use alloc::prelude::v1::Box;
use core::{any::Any, convert::TryInto};

pub trait BlockDevice {
  const NUM_BLOCKS: usize;
  const BLOCK_SIZE: usize;
  fn read(&self, block_num: u32, dst: &mut [u8]) -> Result<usize, ()>;
  fn write(&self, block_num: u32, src: &[u8]) -> Result<usize, ()>;
}

/// Represents Disk Metadata for some application
pub trait Metadata: 'static {
  fn new() -> Self
  where
    Self: Sized;
  fn owned(&self) -> &[u32];

  // TODO possibly add associated error types here

  // TODO these functions need to be purely deterministic, and thus shouldn't contain any unsafe
  // code which allows for raw-ptr manipulation
  // also need to have a functional interface here, to allow for rollback if there are errors.
  fn insert(&self, b: u32) -> Result<Self, ()>
  where
    Self: Sized;

  fn remove(&self, b: u32) -> Result<Self, ()>
  where
    Self: Sized;
}

#[derive(Debug, PartialEq, Eq)]
pub struct BitArray<const N: usize>
where
  [(); { N / 8 }]: , {
  items: [u8; { N / 8 }],
}

impl<const N: usize> BitArray<N>
where
  [(); { N / 8 }]: ,
{
  const fn new(full: bool) -> Self {
    let token = if full { 0xff } else { 0 };
    BitArray {
      items: [token; { N / 8 }],
    }
  }
  /// Sets bit `i` in this bit array.
  pub fn set(&mut self, i: usize) { self.items[i / 8] |= (1 << (i % 8)); }
  /// Unsets bit `i` in this bit array.
  pub fn unset(&mut self, i: usize) { self.items[i / 8] &= !(1 << (i % 8)); }
  /// Gets bit `i` in this bit array, where 1 => true, 0 => false.
  pub fn get(&self, i: usize) -> bool { ((self.items[i / 8] >> (i % 8)) & 1) == 1 }
  /// Iterates through this bit array trying to find a free block, returning none if there are
  /// none.
  pub fn find_free(&self) -> Option<usize> {
    self
      .items
      .iter()
      .enumerate()
      .find(|(_, &v)| v != 0xff)
      .map(|(i, v)| {
        // TODO is this leading or trailing ones?
        i * 8 + v.trailing_ones() as usize
      })
  }
}

// Number of Metadata items stored in this block interface.
const MD_SPACE: usize = 512;
pub struct GlobalBlockInterface<B: BlockDevice>
where
  [(); { B::NUM_BLOCKS / 8 }]: , {
  stored: [Option<Box<dyn Metadata>>; MD_SPACE],

  // TODO below is the number of blocks, just picked an arbitrary number for now
  free_map: BitArray<{ B::NUM_BLOCKS }>,

  // TODO decide whether or not to include a BlockDescriptor sort of thing with an offset?
  block_device: *mut B,
}

use crate::virtio::{Driver, DRIVER};

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub struct MetadataHandle(u32);

pub static mut GLOBAL_BLOCK_INTERFACE: GlobalBlockInterface<Driver> =
  GlobalBlockInterface::new(unsafe { &mut DRIVER });

fn as_dyn(md: &Box<dyn Metadata>) -> &dyn Any { md }

impl<B: BlockDevice> GlobalBlockInterface<B>
where
  [(); { B::NUM_BLOCKS / 8 }]: ,
{
  const fn new(block_device: *mut B) -> Self {
    use core::mem::MaybeUninit;
    let mut stored: [MaybeUninit<Option<Box<dyn Metadata>>>; MD_SPACE] =
      MaybeUninit::uninit_array::<MD_SPACE>();
    let mut i = 0;
    // necessary to use a while loop here because of constraints on const fns.
    while i < MD_SPACE {
      stored[i].write(None);
      i += 1;
    }
    let stored: [Option<_>; MD_SPACE] = unsafe { core::mem::transmute(stored) };
    Self {
      stored,
      free_map: BitArray::new(false),
      block_device,
    }
  }

  /// Allocates new metadata inside of the system
  pub fn new_metadata<M: Metadata>(&mut self) -> Result<MetadataHandle, ()> {
    let md = M::new();
    let starting_owned = md.owned();
    if !starting_owned.is_empty() {
      // We expect metadata to start off empty
      return Err(());
    }
    let free_space = match self.stored.iter_mut().position(|v| v.is_none()) {
      // Out of space
      None => return Err(()),
      Some(space) => space,
    };
    self.stored[free_space] = Some(Box::new(md));
    Ok(MetadataHandle(free_space as u32))
  }

  /// Requests an additional block for a specific MetadataHandle. The allocated block is not
  /// guaranteed to be contiguous.
  pub fn req_block<M: Metadata>(&mut self, MetadataHandle(i): MetadataHandle) -> Result<(), ()> {
    let i = i as usize;
    let md = self.stored.get(i).ok_or(())?;
    let md = md.as_ref().ok_or(())?;
    let mut old_owned = md.owned().iter();

    let new_b = self.free_map.find_free().ok_or(())? as u32;
    let new_md = as_dyn(&md).downcast_ref::<M>().unwrap().insert(new_b)?;

    let mut new_owned = new_md.owned().iter();
    // They must also maintain order between allocations
    if !new_owned
      .by_ref()
      .zip(old_owned.by_ref())
      .all(|(new, prev)| new == prev)
    {
      return Err(());
    }
    match new_owned.next() {
      Some(&b) if b == new_b => {},
      // inserted wrong block
      Some(_) => return Err(()),
      // no block seems to have been inserted
      None => return Err(()),
    };

    if old_owned.next().is_some() {
      // Some block was removed and not kept in the new metadata
      return Err(());
    }
    // perform assignments after all checks
    self.free_map.set(new_b as usize);
    self.stored[i] = Some(Box::new(new_md));

    Ok(())
  }

  /// Reads from `n`th block of the metadata handle into dst
  pub fn read(
    &self,
    MetadataHandle(i): MetadataHandle,
    n: usize,
    dst: &mut [u8],
  ) -> Result<usize, ()> {
    let i = i as usize;
    let md = self.stored.get(i).ok_or(())?;
    let md = md.as_ref().ok_or(())?;
    let owned = md.owned();
    let &b_n = owned.get(n).ok_or(())?;

    unsafe { self.block_device.as_mut().ok_or(())?.read(b_n, dst) }
  }

  /// Writes to the `n`th block of the metadata handle from src
  pub fn write(
    &self,
    MetadataHandle(i): MetadataHandle,
    n: usize,
    src: &[u8],
  ) -> Result<usize, ()> {
    let i = i as usize;
    let md = self.stored.get(i).ok_or(())?;
    let md = md.as_ref().ok_or(())?;
    let owned = md.owned();
    let &b_n = owned.get(n).ok_or(())?;

    unsafe { self.block_device.as_mut().ok_or(())?.write(b_n, src) }
  }
}
