#![no_std]

use core::alloc::Layout;
use core::fmt::{self, Debug, Formatter};
use core::mem::{self, MaybeUninit};
use core::ptr::{self, NonNull};
use core::slice;

/// Utility helping dealing with sizes.
#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq)]
struct Size(usize);

impl Size {
    const fn from_log2(log2: usize) -> Self {
        Self(log2)
    }

    /// Maximum `Size` that fits with in `size` bytes.
    const fn max_size_fit(size: usize) -> Self {
        Self((usize::BITS - 1 - size.leading_zeros()) as usize)
    }

    /// Maximum `Size` that a block starting at `addr` can still naturally align.
    const fn max_addr_align(addr: usize) -> Self {
        Self(addr.trailing_zeros() as usize)
    }

    #[inline]
    fn min_size_layout(layout: Layout) -> Self {
        Self(
            (usize::BITS
                - layout
                    .pad_to_align()
                    .size()
                    .saturating_sub(1)
                    .leading_zeros()) as usize,
        )
    }

    const fn in_log2(self) -> usize {
        self.0
    }

    const fn in_bytes(self) -> usize {
        1 << self.0
    }

    const fn parent(self) -> Size {
        Self(self.0 + 1)
    }

    // SAFETY: Must not be called on the buddy of null pointer.
    unsafe fn buddy_block(self, ptr: NonNull<()>) -> NonNull<()> {
        let addr = (ptr.as_ptr() as usize) ^ (1 << self.0);
        NonNull::new_unchecked(addr as _)
    }

    // SAFETY: Must not be called on a child of null pointer.
    unsafe fn parent_block(self, ptr: NonNull<()>) -> NonNull<()> {
        let addr = (ptr.as_ptr() as usize) & !(1 << self.0);
        NonNull::new_unchecked(addr as _)
    }
}

/// Utility helping dealing with address ranges.
#[derive(Clone, Copy)]
struct AddrRange {
    base: usize,
    size: usize,
}

impl AddrRange {
    fn base_sub(self, size: usize) -> Option<Self> {
        Some(Self {
            base: self.base.checked_sub(size)?,
            size: self.size.checked_add(size)?,
        })
    }

    fn base_add(self, size: usize) -> Option<Self> {
        Some(Self {
            base: self.base.checked_add(size)?,
            size: self.size.checked_sub(size)?,
        })
    }

    fn limit_sub(self, size: usize) -> Option<Self> {
        Some(Self {
            base: self.base,
            size: self.size.checked_sub(size)?,
        })
    }

    fn limit_add(self, size: usize) -> Option<Self> {
        Some(Self {
            base: self.base,
            size: self.size.checked_add(size)?,
        })
    }

    fn base_align_down(self, align: usize) -> Option<Self> {
        let misalignment = self.base & (align - 1);
        self.base_sub(misalignment)
    }

    fn base_align_up(self, align: usize) -> Option<Self> {
        let misalignment = self.base.wrapping_neg() & (align - 1);
        self.base_add(misalignment)
    }

    fn limit_align_down(self, align: usize) -> Option<Self> {
        let misalignment = self.base.wrapping_add(self.size) & (align - 1);
        self.limit_sub(misalignment)
    }

    fn limit_align_up(self, align: usize) -> Option<Self> {
        let misalignment = self.base.wrapping_add(self.size).wrapping_neg() & (align - 1);
        self.limit_add(misalignment)
    }

    fn align_expand(self, align: usize) -> Option<Self> {
        self.base_align_down(align)?.limit_align_up(align)
    }
}

/// A bitmap backed by `[u8]` slice.
#[repr(transparent)]
struct Bitmap([u8]);

impl Bitmap {
    /// Get size in bytes needed for at least given number of entries.
    const fn size_needed(entries: usize) -> usize {
        (entries + 7) / 8
    }

    /// Create a bitmap from a slice.
    fn new(slice: &mut [u8]) -> &mut Self {
        unsafe { mem::transmute(slice) }
    }

    /// Test if a bit is set.
    fn test(&self, index: usize) -> bool {
        let byte = &self.0[index / 8];
        let mask = 1 << (index % 8);
        *byte & mask != 0
    }

    /// Flip a bit and return the original.
    fn flip(&mut self, index: usize) -> bool {
        let byte = &mut self.0[index / 8];
        let mask = 1 << (index % 8);
        *byte ^= mask;
        *byte & mask == 0
    }
}

struct FreeBlockNode {
    prev: Option<NonNull<FreeBlockNode>>,
    next: Option<NonNull<FreeBlockNode>>,
}

struct FreeBlockList {
    head: Option<NonNull<FreeBlockNode>>,
}

unsafe impl Send for FreeBlockList {}

impl FreeBlockList {
    // Insert the block into the head of the linked list.
    unsafe fn push_block(&mut self, block_ptr: NonNull<()>) {
        let mut block_ptr = block_ptr.cast::<FreeBlockNode>();
        let block = block_ptr.as_mut();
        block.prev = None;
        block.next = self.head;

        // Update first.prev
        match self.head {
            // End of list
            None => (),
            Some(mut next_block) => {
                let next_block = next_block.as_mut();
                next_block.prev = Some(block_ptr);
            }
        }

        // Update first
        self.head = Some(block_ptr);
    }

    // Remove a block from the head of the linked list.
    fn pop_block(&mut self) -> Option<NonNull<()>> {
        unsafe {
            let mut block_ptr = match self.head {
                None => return None,
                Some(head) => head,
            };
            let block = block_ptr.as_mut();
            self.head = block.next;

            // Update next.prev
            match block.next {
                // End of list
                None => (),
                Some(mut next_block) => {
                    let next_block = next_block.as_mut();
                    next_block.prev = None;
                }
            }

            Some(block_ptr.cast())
        }
    }

    // Remove block from the linked list
    unsafe fn remove_block(&mut self, block_ptr: NonNull<()>) {
        let mut block_ptr = block_ptr.cast::<FreeBlockNode>();
        let block = block_ptr.as_mut();

        // Update prev.next
        match block.prev {
            // Start of the list
            None => {
                self.head = block.next;
            }
            Some(mut prev_block) => {
                let prev_block = prev_block.as_mut();
                prev_block.next = block.next;
            }
        }

        // Update next.prev
        match block.next {
            // End of list
            None => (),
            Some(mut next_block) => {
                let next_block = next_block.as_mut();
                next_block.prev = block.prev;
            }
        }
    }
}

impl Debug for FreeBlockList {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_list();
        let mut opt_ptr = self.head;
        while let Some(mut ptr) = opt_ptr {
            debug.entry(&ptr);
            opt_ptr = unsafe { ptr.as_mut() }.next;
        }
        debug.finish()
    }
}

struct BuddyGroup<'a> {
    free_blocks: FreeBlockList,
    bitmap: &'a mut Bitmap,
    base: usize,
}

impl Debug for BuddyGroup<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("BuddyGroup")
            .field("free_blocks", &self.free_blocks)
            .field("base", &self.base)
            .finish()
    }
}

impl BuddyGroup<'_> {
    fn allocate(&mut self, size: Size) -> Option<NonNull<()>> {
        let block = self.free_blocks.pop_block()?;

        // Flip the bit in the bitmap.
        // Note that bitmap records a single bit for two consecutive blocks.
        let bitmap_index =
            (block.as_ptr() as usize).checked_sub(self.base).unwrap() >> size.parent().in_log2();
        self.bitmap.flip(bitmap_index);

        Some(block)
    }

    unsafe fn deallocate(&mut self, size: Size, block: NonNull<()>) -> Option<NonNull<()>> {
        // Flip the bit in the bitmap.
        let bitmap_index =
            (block.as_ptr() as usize).checked_sub(self.base).unwrap() >> size.parent().in_log2();

        // If the bit is original set, then one of the blocks are free.
        // Since we know that the current one is not free, so the other one must be free.
        if self.bitmap.flip(bitmap_index) {
            self.free_blocks.remove_block(size.buddy_block(block));
            return Some(size.parent_block(block));
        }

        self.free_blocks.push_block(block);
        None
    }

    // Try to find if buddy is allocated for an already-allocated `block`.
    unsafe fn is_buddy_allocated(&mut self, size: Size, block: NonNull<()>) -> bool {
        // Flip the bit in the bitmap.
        let bitmap_index =
            (block.as_ptr() as usize).checked_sub(self.base).unwrap() >> size.parent().in_log2();

        // If the bit is set, then one of the blocks are free.
        // Since we know that the current one is not free, so the other one must be free.
        !self.bitmap.test(bitmap_index)
    }
}

const MIN_BLOCK: usize = 4; // 16B
const MAX_BLOCK: usize = 30; // 1GiB
const _: () = assert!((1 << MIN_BLOCK) >= mem::size_of::<FreeBlockNode>());

#[repr(C)]
struct BuddyGroups<'a> {
    groups: Option<&'a mut [BuddyGroup<'a>; MAX_BLOCK - MIN_BLOCK]>,
    catch_all: FreeBlockList,
}

impl Debug for BuddyGroups<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_map();
        for i in MIN_BLOCK..MAX_BLOCK {
            debug.entry(&i, &self.groups.as_ref().unwrap()[i - MIN_BLOCK]);
        }
        debug.entry(&MAX_BLOCK, &self.catch_all).finish()
    }
}

impl BuddyGroups<'_> {
    #[cold]
    fn allocate_max(&mut self, size: Size) -> Option<NonNull<()>> {
        // Too large to allocate
        if size != Size::from_log2(MAX_BLOCK) {
            return None;
        }

        self.catch_all.pop_block()
    }

    #[cold]
    unsafe fn deallocate_max(&mut self, size: Size, ptr: NonNull<()>) {
        for i in 0..(1 << (size.0 - MAX_BLOCK)) {
            let block = ptr.as_ptr() as usize + i + Size::from_log2(MAX_BLOCK).in_bytes();
            self.catch_all
                .push_block(NonNull::new_unchecked(block as _));
        }
    }

    fn allocate(&mut self, size: Size) -> Option<NonNull<()>> {
        if size >= Size::from_log2(MAX_BLOCK) {
            return self.allocate_max(size);
        }

        let size = size.max(Size::from_log2(MIN_BLOCK));
        let group = self.groups.as_mut()?.get_mut(size.in_log2() - MIN_BLOCK)?;
        match group.allocate(size) {
            Some(v) => Some(v),
            None => {
                let ptr = self.allocate(size.parent())?;
                unsafe {
                    self.groups.as_mut()?[size.in_log2() - MIN_BLOCK]
                        .deallocate(size, size.buddy_block(ptr))
                        .map(|_| unreachable!());
                };
                Some(ptr)
            }
        }
    }

    unsafe fn deallocate(&mut self, size: Size, ptr: NonNull<()>) {
        if size >= Size::from_log2(MAX_BLOCK) {
            return self.deallocate_max(size, ptr);
        }

        let size = size.max(Size::from_log2(MIN_BLOCK));
        match self.groups.as_mut().unwrap()[size.in_log2() - MIN_BLOCK].deallocate(size, ptr) {
            None => (),
            Some(v) => {
                self.deallocate(size.parent(), v);
            }
        }
    }

    unsafe fn shrink(&mut self, size: Size, mut new_size: Size, block: NonNull<()>) {
        while new_size < size {
            self.deallocate(new_size, new_size.buddy_block(block));
            new_size = new_size.parent();
        }
    }

    // Try to grow `block` in place.
    unsafe fn grow(&mut self, mut size: Size, new_size: Size, block: NonNull<()>) -> bool {
        // Check for alignment, if unaligned then this is definitely not possible
        let aligned_ptr = block.as_ptr() as usize >> new_size.in_log2() << new_size.in_log2();
        if block.as_ptr() as usize != aligned_ptr {
            return false;
        }

        let mut test_size = size;
        while test_size < new_size {
            if self.groups.as_mut().unwrap()[test_size.in_log2() - MIN_BLOCK]
                .is_buddy_allocated(test_size, block)
            {
                return false;
            }
            test_size = test_size.parent()
        }

        while size < new_size {
            let v =
                self.groups.as_mut().unwrap()[size.in_log2() - MIN_BLOCK].deallocate(size, block);
            assert_eq!(v, Some(block));
            size = size.parent();
        }

        true
    }

    fn new(memory: &mut [u8]) -> Option<Self> {
        let mut range = AddrRange {
            base: memory.as_ptr() as usize,
            size: memory.len(),
        };

        range = range
            .base_align_up(mem::align_of::<BuddyGroup>())?
            .limit_align_down(1 << MIN_BLOCK)?;

        // Reserve some memory for `BuddyGroup`s.
        let levels_addr = range.base;
        range = range.base_add(mem::size_of::<[BuddyGroup; MAX_BLOCK - MIN_BLOCK]>())?;
        let groups: &mut [MaybeUninit<BuddyGroup>; MAX_BLOCK - MIN_BLOCK] =
            unsafe { &mut *(levels_addr as *mut _) };

        for i in MIN_BLOCK..MAX_BLOCK {
            let block_size = Size::from_log2(i);

            // Reserve some memory for `Bitmap`.
            let range_for_bitmap = range.align_expand(block_size.parent().in_bytes())?;
            let bitmap_addr = range.base as *mut u8;
            let bitmap_len =
                Bitmap::size_needed(range_for_bitmap.size >> block_size.parent().in_log2());
            range = range.base_add(bitmap_len)?;

            // Initialize bitmap
            unsafe { ptr::write_bytes(bitmap_addr, 0, bitmap_len) };
            let bitmap_u8 = unsafe { slice::from_raw_parts_mut(bitmap_addr, bitmap_len) };
            let bitmap = Bitmap::new(bitmap_u8);

            groups[i - MIN_BLOCK].write(BuddyGroup {
                free_blocks: FreeBlockList { head: None },
                bitmap,
                base: range_for_bitmap.base,
            });
        }

        let groups = unsafe { mem::transmute(groups) };
        let mut alloc = BuddyGroups {
            groups: Some(groups),
            catch_all: FreeBlockList { head: None },
        };

        range = range.base_align_up(1 << MIN_BLOCK)?;
        while range.size != 0 {
            let size = Size::max_addr_align(range.base).min(Size::max_size_fit(range.size));
            unsafe { alloc.deallocate(size, NonNull::new_unchecked(range.base as _)) };
            range = range.base_add(size.in_bytes())?;
        }

        Some(alloc)
    }

    const fn empty() -> Self {
        #[repr(C)]
        struct Dummy {
            groups: Option<&'static [BuddyGroup<'static>; MAX_BLOCK - MIN_BLOCK]>,
            catch_all: FreeBlockList,
        }
        unsafe {
            mem::transmute(Dummy {
                groups: None,
                catch_all: FreeBlockList { head: None },
            })
        }
    }
}

pub struct BuddyAllocator<'a>(spin::Mutex<BuddyGroups<'a>>);

#[derive(Debug, Clone, Copy)]
pub enum InitError {
    AlreadyInitialized,
    MemoryTooSmall,
}

impl<'a> BuddyAllocator<'a> {
    pub const fn new() -> Self {
        Self(spin::Mutex::new(BuddyGroups::empty()))
    }

    pub fn initialize(&self, memory: &'a mut [u8]) -> Result<(), InitError> {
        let mut guard = self.0.lock();
        if !guard.groups.is_none() {
            return Err(InitError::AlreadyInitialized);
        }
        *guard = BuddyGroups::new(memory).ok_or(InitError::MemoryTooSmall)?;
        Ok(())
    }
}

unsafe impl core::alloc::GlobalAlloc for BuddyAllocator<'_> {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() == 0 {
            return layout.align() as _;
        }
        let size = Size::min_size_layout(layout);
        match self.0.lock().allocate(size) {
            None => ptr::null_mut(),
            Some(v) => v.as_ptr() as _,
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let size = Size::min_size_layout(layout);
        self.0
            .lock()
            .deallocate(size, NonNull::new_unchecked(ptr as *mut _))
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_layout = Layout::from_size_align_unchecked(new_size, layout.align());
        let size = Size::min_size_layout(layout);
        let new_size = Size::min_size_layout(new_layout);
        if size == new_size {
            ptr
        } else if size > new_size {
            self.0
                .lock()
                .shrink(size, new_size, NonNull::new_unchecked(ptr as *mut _));
            ptr
        } else {
            let mut guard = self.0.lock();
            if guard.grow(size, new_size, NonNull::new_unchecked(ptr as *mut _)) {
                ptr
            } else {
                let new_ptr = match guard.allocate(new_size) {
                    None => return ptr::null_mut(),
                    Some(v) => v,
                }
                .as_ptr() as *mut u8;
                drop(guard);
                ptr::copy_nonoverlapping(ptr, new_ptr, layout.size());
                self.0
                    .lock()
                    .deallocate(size, NonNull::new_unchecked(ptr as *mut _));
                new_ptr
            }
        }
    }
}
