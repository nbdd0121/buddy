use std::alloc::{GlobalAlloc, Layout};

use buddy::BuddyAllocator;
use once_cell::sync::Lazy;

static ALLOC: Lazy<BuddyAllocator<'static>> = Lazy::new(|| {
    // Allocate 1GiB
    let memory = unsafe {
        libc::mmap(
            core::ptr::null_mut(),
            1 << 30,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    assert_ne!(memory, libc::MAP_FAILED);
    BuddyAllocator::new(unsafe { core::slice::from_raw_parts_mut(memory as *mut u8, 1 << 30) })
        .unwrap()
});

struct Shim();

#[global_allocator]
static G: Shim = Shim();

unsafe impl GlobalAlloc for Shim {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        ALLOC.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOC.realloc(ptr, layout, new_size)
    }
}

fn main() {
    unsafe {
        let a = ALLOC.alloc(Layout::new::<u32>());
        println!("{:p}", a);
        let a = ALLOC.alloc(Layout::new::<u32>());
        println!("{:p}", a);
    }
    let a = vec![1, 2, 3];
    println!("{:p}", a.as_ptr());
}
