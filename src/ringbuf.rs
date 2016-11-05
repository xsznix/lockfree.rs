use std::intrinsics::{
  atomic_cxchg,
  atomic_xadd,
  atomic_xsub,
  volatile_load,
  volatile_store,
};

use std::boxed::Box;
use std::clone::Clone;
use std::marker::Send;
use std::ops::{Deref, DerefMut, Drop};
use std::option::Option;
use std::ptr::Shared;
use std::thread;

/// A lock-free ring buffer, inspired by the following:
/// http://www.codeproject.com/Articles/153898/Yet-another-implementation-of-a-lock-free-circul
///
/// In addition to the write index, read index, and max read index, we also store a max write index
/// in order to prevent producers from writing over data that has been dequeued but not yet read.
/// Otherwise, the implementation is equivalent to a literal translation of the C++ code in the
/// linked article.
pub struct RingBuf<T> {
  write_index: usize,
  read_index: usize,
  max_write_index: usize,
  max_read_index: usize,
  buf: Box<[T]>,
}

impl <T> RingBuf<T> {
  /// Creates a new ring buffer of the specified length.
  ///
  /// In order to be correct after more than `usize::max_value()` reads, `len` should be a power of
  /// 2. This is because all of the indices into the buffer are interpreted modulo `len`, so when
  /// the indices wrap around, if `len` does not divide `usize::max_value()` they may no longer be
  /// valid.
  pub fn new(len: usize) -> RingBuf<T> {
    // We don't need to fully initialize the buffer, since we promise not to read any values that
    // have not been written previously.
    let mut buf: Vec<T> = Vec::with_capacity(len);
    unsafe { buf.set_len(len); }
    RingBuf {
      write_index: 0,
      read_index: 0,
      max_write_index: len - 1,
      max_read_index: 0,
      buf: buf.into_boxed_slice(),
    }
  }

  /// Pushes/enqueues the specified value onto the ring buffer. Returns `true` if the operation
  /// succeeded and `false` otherwise.
  ///
  /// The operation can fail if the queue is full, i.e. there are no open spaces in the backing
  /// array or there will be soon but some readers have not yet committed reads/pops/dequeues.
  pub fn push(&mut self, item: T) -> bool {
    unsafe {
      let mut current_write_index: usize;
      let mut max_write_index: usize;

      // Reserve a space.
      loop {
        current_write_index = volatile_load(&self.write_index as *const usize);
        max_write_index = volatile_load(&self.max_write_index as *const usize);
        if current_write_index == max_write_index + 1 {
          // The queue is full or has an uncommitted read.
          return false;
        }

        // If no other thread has successfully reserved a space to write to between the start of
        // this loop iteration and here, this compare-and-swap will succeed and we will have
        // officially reserved our space.
        let (_, success) = atomic_cxchg(
            &mut self.write_index as *mut usize,
            current_write_index,
            current_write_index.wrapping_add(1));
        if success {
          break;
        }
      }

      // Now that we have reserved a space, save the data.
      volatile_store(&mut self.buf[self.count_to_index(current_write_index)] as *mut T, item);

      // Update the maximum read index so that consumers can read the data we just wrote.
      loop {
        let (_, success) = atomic_cxchg(
            &mut self.max_read_index as *mut usize,
            current_write_index,
            current_write_index.wrapping_add(1));
        if success {
          break;
        }

        // If the exchange failed, then some other thread must be also pushing at the same time as
        // us. Yield so we avoid busy waiting.
        thread::yield_now();
      }

      true
    }
  }

  /// Pops/dequeues the next item from the queue and returns it.
  ///
  /// The operation may fail and return `None` if the queue is empty or if the queue will have items
  /// soon but some producers have not yet committed their writes/pushes/enqueues.
  pub fn pop(&mut self) -> Option<T> {
    unsafe {
      let mut current_max_read_index: usize;
      let mut current_read_index: usize;

      // Reserve a space to read from.
      loop {
        current_read_index = volatile_load(&self.read_index as *const usize);
        current_max_read_index = volatile_load(&self.max_read_index as *const usize);

        if current_read_index == current_max_read_index {
          // The queue is empty or a producer has an uncommitted write.
          return None;
        }

        // If no other thread has successfully reserved a space to read from between the start of
        // this loop iteration and here, this compare-and-swap will succeed and we will have
        // officially reserved our space.
        let (_, success) = atomic_cxchg(
          &mut self.read_index as *mut usize,
          current_read_index,
          current_read_index.wrapping_add(1));
        if success {
          break;
        }
      }

      // We've successfully obtained an item. Read it, and only then allow it to be written over.
      let item = volatile_load(&self.buf[self.count_to_index(current_read_index)] as *const T);

      // Update the maximum write index so that we can reuse this space.
      loop {
        let (_, success) = atomic_cxchg(
          &mut self.max_write_index as *mut usize,
          current_read_index.wrapping_add(self.buf.len() - 1),
          current_read_index.wrapping_add(self.buf.len()));
        if success {
          break;
        }

        // Some other thread is also popping right now. Yield to avoid busy waiting.
        thread::yield_now();
      }

      Some(item)
    }
  }

  #[inline]
  fn count_to_index(&self, count: usize) -> usize {
    count % self.buf.len()
  }
}

/// Points to a ring buffer. When cloned, the clone will point to the same ringbuffer. Thus, to
/// share a ring buffer safely between threads, create a `SharedRingBuf` and make a clone for each
/// thread that needs to use it. Each instance of `SharedRingBuf` will increment the reference count
/// to the shared `RingBuf` instance and the backing instance will be dropped once the last instance
/// is dropped.
///
/// The methods `push` and `pop` of the ring buffer are exposed by `SharedRingBuf` through the
/// `Deref` and `DerefMut` traits.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// let mut buf = SharedRingBuf::new(1024);
/// buf.push(1);
/// buf.push(2);
/// buf.push(3);
///
/// let mut buf_copy = buf.clone();
/// let guard = thread::spawn(move || {
///   println!("{}", buf_copy.pop().unwrap()); // 1
///   println!("{}", buf_copy.pop().unwrap()); // 2
///   println!("{}", buf_copy.pop().unwrap()); // 3
/// });
/// let _ = guard.join();
pub struct SharedRingBuf<T> {
  refcount: Shared<usize>,
  buf: Shared<RingBuf<T>>,
}

impl<T> SharedRingBuf<T> {
  pub fn new(len: usize) -> SharedRingBuf<T> {
    unsafe {
      SharedRingBuf {
        refcount: Shared::new(Box::into_raw(Box::new(1))),
        buf: Shared::new(Box::into_raw(Box::new(RingBuf::new(len)))),
      }
    }
  }
}

impl<T> Deref for SharedRingBuf<T> {
  type Target = RingBuf<T>;

  fn deref(&self) -> &RingBuf<T> {
    // Safe because buf is never null.
    unsafe { self.buf.as_ref().unwrap() }
  }
}

impl<T> DerefMut for SharedRingBuf<T> {
  fn deref_mut(&mut self) -> &mut RingBuf<T> {
    // Safe because buf is never null.
    unsafe { self.buf.as_mut().unwrap() }
  }
}

impl<T> Clone for SharedRingBuf<T> {
  fn clone(&self) -> SharedRingBuf<T> {
    unsafe { atomic_xadd(self.refcount.as_mut().unwrap(), 1usize); }

    SharedRingBuf {
      refcount: self.refcount.clone(),
      buf: self.buf.clone(),
    }
  }
}

impl<T> Drop for SharedRingBuf<T> {
  fn drop(&mut self) {
    unsafe {
      let refs = atomic_xsub(self.refcount.as_mut().unwrap(), 1usize);
      if refs == 1 {
        let _ = Box::from_raw(self.refcount.as_mut().unwrap());
        let _ = Box::from_raw(self.buf.as_mut().unwrap());
      }
    }
  }
}

/// Send needs to be "implemented" in order for a `SharedRingBuf` to be sent to another thread. This
/// is safe because a lock-free ring buffer is by definition thread-safe if the implementation is
/// correct.
unsafe impl<T> Send for SharedRingBuf<T> {}

#[cfg(test)]
mod tests {
  use super::{RingBuf, SharedRingBuf};
  use std::thread;

  #[test]
  fn single_thread() {
    let mut buf = RingBuf::new(100);

    for i in 0..100 {
      assert!(buf.push(i));
    }

    for i in 0..100 {
      match buf.pop() {
        Some(item) => assert_eq!(item, i),
        None => assert!(false),
      }
    }
  }

  #[test]
  fn single_thread_two_pass() {
    let mut buf = RingBuf::new(100);

    for i in 0..100 {
      assert!(buf.push(i));
    }

    for i in 0..50 {
      match buf.pop() {
        Some(item) => assert_eq!(item, i),
        None => assert!(false),
      }
    }

    for i in 100..150 {
      assert!(buf.push(i));
    }

    for i in 50..150 {
      match buf.pop() {
        Some(item) => assert_eq!(item, i),
        None => assert!(false),
      }
    }
  }

  #[test]
  fn concurrent_consumer() {
    let mut buf = SharedRingBuf::new(10000000);

    // Push 10 million consecutive numbers.
    for i in 0..10000000 {
      assert!(buf.push(i));
    }

    // Create two consumers to read all of the numbers concurrently.
    let consumer_guard_1 = {
      let mut buf = buf.clone();
      let mut results = Vec::with_capacity(5000000);
      thread::spawn(move || {
        let mut last = 0;
        for _ in 0..5000000 {
          match buf.pop() {
            Some(item) => {
              if last != 0 {
                assert!(item > last);
              }
              last = item;
              results.push(item);
            },
            None => assert!(false),
          }
        }

        results
      })
    };

    let consumer_guard_2 = {
      let mut buf = buf.clone();
      let mut results = Vec::with_capacity(5000000);
      thread::spawn(move || {
        let mut last = 0;
        for _ in 0..5000000 {
          match buf.pop() {
            Some(item) => {
              if last != 0 {
                assert!(item > last);
              }
              last = item;
              results.push(item);
            },
            None => assert!(false),
          }
        }

        results
      })
    };

    // Check that the consumers got the numbers in the right order and without
    // skips or duplicates.
    let res1 = consumer_guard_1.join().unwrap();
    let res2 = consumer_guard_2.join().unwrap();
    let mut i = 0;
    let mut j = 0;
    for total in 0..10000000 {
      if i == 5000000 {
        if res2[j] == total {
          j += 1;
        } else {
          assert!(false);
        }
      } else if j == 5000000 {
        if res1[i] == total {
          i += 1;
        } else {
          assert!(false);
        }
      } else {
        let a = res1[i];
        let b = res2[j];
        if a == b {
          assert!(false);
        } else if a < b {
          if a == total {
            i += 1;
          } else {
            assert!(false);
          }
        } else {
          if b == total {
            j += 1;
          } else {
            assert!(false);
          }
        }
      }
    }

    // Check that there are no more items in the buffer.
    assert_eq!(buf.pop(), None);
  }
}
