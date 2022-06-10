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

const MIN_BLOCK: usize = 4; // 16B
const MAX_BLOCK: usize = 30; // 1GiB
const _: () = assert!((1 << MIN_BLOCK) >= mem::size_of::<FreeBlockNode>());

#[repr(C)]
struct BuddyGroups<'a> {
    range: AddrRange,
    free_blocks: [FreeBlockList; MAX_BLOCK - MIN_BLOCK],
    metadata: [(&'a mut Bitmap, usize); MAX_BLOCK - MIN_BLOCK],
    catch_all: FreeBlockList,
}

impl Debug for BuddyGroups<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_map();
        for i in MIN_BLOCK..MAX_BLOCK {
            debug.entry(&i, &self.free_blocks[i - MIN_BLOCK]);
        }
        debug.entry(&MAX_BLOCK, &self.catch_all).finish()
    }
}

impl BuddyGroups<'_> {
    // Test if a bit is set.
    fn metadata_test(&self, size: Size, ptr: usize) -> bool {
        debug_assert!((MIN_BLOCK..MAX_BLOCK).contains(&size.in_log2()));

        let (ref bitmap, base) = self.metadata[size.in_log2() - MIN_BLOCK];
        let bitmap_index = ptr.checked_sub(base).unwrap() >> size.parent().in_log2();
        bitmap.test(bitmap_index)
    }

    // Flip a bit and return the original.
    fn metadata_flip(&mut self, size: Size, offset: usize) -> bool {
        debug_assert!((MIN_BLOCK..MAX_BLOCK).contains(&size.in_log2()));

        let (ref mut bitmap, base) = self.metadata[size.in_log2() - MIN_BLOCK];
        let bitmap_index = offset.checked_sub(base).unwrap() >> size.parent().in_log2();
        bitmap.flip(bitmap_index)
    }

    fn allocate_exact(&mut self, size: Size) -> Option<NonNull<()>> {
        debug_assert!((MIN_BLOCK..MAX_BLOCK).contains(&size.in_log2()));

        let block = self.free_blocks[size.in_log2() - MIN_BLOCK].pop_block()?;
        self.metadata_flip(size, block.as_ptr() as usize);

        Some(block)
    }

    unsafe fn deallocate_exact(&mut self, size: Size, block: NonNull<()>) -> Option<NonNull<()>> {
        debug_assert!((MIN_BLOCK..MAX_BLOCK).contains(&size.in_log2()));

        // If the bit is original set, then one of the blocks are free.
        // Since we know that the current one is not free, so the other one must be free.
        if self.metadata_flip(size, block.as_ptr() as usize) {
            self.free_blocks[size.in_log2() - MIN_BLOCK].remove_block(size.buddy_block(block));
            return Some(size.parent_block(block));
        }

        self.free_blocks[size.in_log2() - MIN_BLOCK].push_block(block);
        None
    }

    // Try to find if buddy is allocated for an already-allocated `block`.
    unsafe fn is_buddy_allocated(&mut self, size: Size, block: NonNull<()>) -> bool {
        debug_assert!((MIN_BLOCK..MAX_BLOCK).contains(&size.in_log2()));

        // If the bit is set, then one of the blocks are free.
        // Since we know that the current one is not free, so the other one must be free.
        !self.metadata_test(size, block.as_ptr() as usize)
    }

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
        match self.allocate_exact(size) {
            Some(v) => Some(v),
            None => {
                let ptr = self.allocate(size.parent())?;
                unsafe {
                    self.deallocate_exact(size, size.buddy_block(ptr))
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
        match self.deallocate_exact(size, ptr) {
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

    // Check if `block` can grow in place.
    unsafe fn can_grow(&mut self, size: Size, new_size: Size, block: NonNull<()>) -> bool {
        // Check for alignment, if unaligned then this is definitely not possible
        let aligned_ptr = block.as_ptr() as usize >> new_size.in_log2() << new_size.in_log2();
        if block.as_ptr() as usize != aligned_ptr {
            return false;
        }

        let mut test_size = size;
        while test_size < new_size {
            if self.is_buddy_allocated(test_size, block) {
                return false;
            }
            test_size = test_size.parent()
        }

        true
    }

    // Commit the block growth.
    unsafe fn grow(&mut self, mut size: Size, new_size: Size, block: NonNull<()>) {
        debug_assert!(self.can_grow(size, new_size, block));

        while size < new_size {
            let v = self.deallocate_exact(size, block);
            assert_eq!(v, Some(block));
            size = size.parent();
        }
    }

    unsafe fn add_memory(&mut self, mut memory: AddrRange) -> Option<()> {
        memory = memory.base_align_up(1 << MIN_BLOCK)?;
        while memory.size != 0 {
            let size = Size::max_addr_align(memory.base).min(Size::max_size_fit(memory.size));
            self.deallocate(size, NonNull::new_unchecked(memory.base as _));
            memory = memory.base_add(size.in_bytes())?;
        }
        Some(())
    }

    unsafe fn new(mut range: AddrRange, mut memory: AddrRange) -> Option<Self> {
        range = range
            .base_align_up(1 << MIN_BLOCK)?
            .limit_align_down(1 << MIN_BLOCK)?;

        let mut metadata: [MaybeUninit<(&mut Bitmap, usize)>; MAX_BLOCK - MIN_BLOCK] =
            MaybeUninit::uninit().assume_init();

        for i in MIN_BLOCK..MAX_BLOCK {
            let block_size = Size::from_log2(i);

            // Reserve some memory for `Bitmap`.
            let range_for_bitmap = range.align_expand(block_size.parent().in_bytes())?;
            let bitmap_addr = memory.base as *mut u8;
            let bitmap_len =
                Bitmap::size_needed(range_for_bitmap.size >> block_size.parent().in_log2());
            memory = memory.base_add(bitmap_len)?;

            // Initialize bitmap
            ptr::write_bytes(bitmap_addr, 0, bitmap_len);
            let bitmap_u8 = slice::from_raw_parts_mut(bitmap_addr, bitmap_len);
            let bitmap = Bitmap::new(bitmap_u8);

            metadata[i - MIN_BLOCK].write((bitmap, range_for_bitmap.base));
        }

        const FREE_LIST: FreeBlockList = FreeBlockList { head: None };

        let mut alloc = BuddyGroups {
            range,
            free_blocks: [FREE_LIST; MAX_BLOCK - MIN_BLOCK],
            metadata: mem::transmute(metadata),
            catch_all: FreeBlockList { head: None },
        };

        alloc.add_memory(memory);
        Some(alloc)
    }
}

/// Buddy allocator.
pub struct BuddyAllocator<'a>(spin::Mutex<BuddyGroups<'a>>);

/// Initialization failure.
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub enum InitError {
    MemoryTooSmall,
}

/// Allocation failure.
#[non_exhaustive]
pub struct AllocError;

fn nonnull_slice(ptr: NonNull<()>, size: Size) -> NonNull<[u8]> {
    unsafe {
        NonNull::new_unchecked(slice::from_raw_parts_mut(
            ptr.as_ptr() as _,
            size.in_bytes(),
        ))
    }
}

impl<'a> BuddyAllocator<'a> {
    /// Create a buddy allocator given a free memory region.
    ///
    /// Unlike [`Self::with_range`], memory outside the supplied `memory` region cannot be added
    /// later.
    pub fn new(memory: &'a mut [u8]) -> Result<Self, InitError> {
        Self::with_range(memory.as_ptr() as usize, memory.len(), memory)
    }

    /// Create a buddy allocator that can handle the given range.
    ///
    /// `memory` is an initial region within the given range that is free. Part of it will be used
    /// for metadata, and the rest will be available for allocation.
    ///
    /// Available memory can be added further later.
    pub fn with_range(base: usize, size: usize, memory: &'a mut [u8]) -> Result<Self, InitError> {
        let range = AddrRange { base, size };
        let memory = AddrRange {
            base: memory.as_ptr() as usize,
            size: memory.len(),
        };

        assert!(range.base <= memory.base);
        // Prevents overflow.
        let residual = range.size - (memory.base - range.base);
        assert!(memory.size <= residual);

        Ok(BuddyAllocator(spin::Mutex::new(
            unsafe { BuddyGroups::new(range, memory) }.ok_or(InitError::MemoryTooSmall)?,
        )))
    }

    /// Add free memory to the allocator.
    pub fn add_memory(&mut self, memory: &'a mut [u8]) {
        let mut alloc = self.0.lock();
        let memory = AddrRange {
            base: memory.as_ptr() as usize,
            size: memory.len(),
        };

        // Prevents overflow.
        assert!(alloc.range.base <= memory.base);
        let residual = alloc.range.size - (memory.base - alloc.range.base);
        assert!(memory.size <= residual);

        unsafe {
            alloc.add_memory(memory);
        }
    }

    /// Attempts to allocate a block of memory.
    pub fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            return Ok(unsafe {
                NonNull::new_unchecked(slice::from_raw_parts_mut(layout.align() as _, 0))
            });
        }
        let size = Size::min_size_layout(layout);
        let ptr = self.0.lock().allocate(size).ok_or(AllocError)?;
        Ok(nonnull_slice(ptr, size))
    }

    /// Deallocates the memory referenced by `ptr`.
    pub unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let size = Size::min_size_layout(layout);
        self.0
            .lock()
            .deallocate(size, NonNull::new_unchecked(ptr.as_ptr() as *mut _))
    }

    /// Attempts to extend the memory block.
    pub unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let ptr: NonNull<()> = ptr.cast();
        let old_size = Size::min_size_layout(old_layout);
        let new_size = Size::min_size_layout(new_layout);
        if old_size == new_size {
            return Ok(nonnull_slice(ptr, new_size));
        }

        let mut guard = self.0.lock();
        if guard.can_grow(old_size, new_size, ptr) {
            guard.grow(old_size, new_size, ptr);
            Ok(nonnull_slice(ptr, new_size))
        } else {
            let new_ptr = guard.allocate(new_size).ok_or(AllocError)?;
            drop(guard);
            ptr::copy_nonoverlapping(
                ptr.as_ptr() as *const u8,
                new_ptr.as_ptr() as *mut u8,
                old_layout.size(),
            );
            self.0.lock().deallocate(old_size, ptr);
            Ok(nonnull_slice(new_ptr, new_size))
        }
    }

    /// Attempts to shrink the memory block.
    pub unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let ptr: NonNull<()> = ptr.cast();
        let old_size = Size::min_size_layout(old_layout);
        let new_size = Size::min_size_layout(new_layout);
        if old_size == new_size {
            return Ok(nonnull_slice(ptr, new_size));
        }

        self.0.lock().shrink(old_size, new_size, ptr);
        Ok(nonnull_slice(ptr, new_size))
    }
}

unsafe impl core::alloc::GlobalAlloc for BuddyAllocator<'_> {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        match self.allocate(layout) {
            Ok(slice) => (*slice.as_ptr()).as_mut_ptr(),
            Err(_) => core::ptr::null_mut(),
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.deallocate(NonNull::new_unchecked(ptr), layout);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_layout = Layout::from_size_align_unchecked(new_size, layout.align());
        let slice = if layout.size() >= new_size {
            self.shrink(NonNull::new_unchecked(ptr), layout, new_layout)
        } else {
            self.grow(NonNull::new_unchecked(ptr), layout, new_layout)
        };
        match slice {
            Ok(slice) => (*slice.as_ptr()).as_mut_ptr(),
            Err(_) => core::ptr::null_mut(),
        }
    }
}
