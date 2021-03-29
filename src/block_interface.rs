use crate::{
  bit_array::{nearest_div_8, BitArray},
  virtio::{Driver, DRIVER},
};
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

  // TODO maybe add a vectorized version of below instead of allocating one at a time which may
  // be costly?

  // TODO these functions need to be purely deterministic, and thus shouldn't contain any unsafe
  // code which allows for raw-ptr manipulation
  // also need to have a functional interface here, to allow for rollback if there are errors.
  fn insert(&self, b: u32) -> Result<Self, ()>
  where
    Self: Sized;

  fn remove(&self, b: u32) -> Result<Self, ()>
  where
    Self: Sized;

  fn ser(&self) -> &[u8]
  where
    Self: Sized, {
    unsafe {
      core::ptr::slice_from_raw_parts(
        (self as *const Self) as *const u8,
        core::mem::size_of::<Self>(),
      )
      .as_ref()
      .unwrap()
    }
  }
  /// returns the number of bytes read and an instance of Self
  fn de(bytes: &[u8]) -> Result<(Self, usize), ()>
  where
    Self: Sized, {
    let len = core::mem::size_of::<Self>();
    let used = &bytes.get(..len).ok_or(())?;
    assert!(used.len() == len);
    let v = unsafe { (used.as_ptr() as *const Self).read() };
    Ok((v, len))
  }
}

// Number of Metadata items stored in this block interface.
const MD_SPACE: usize = 512;
// Metadata currently has no owner
const NO_OWNER: u8 = u8::MAX;
pub struct GlobalBlockInterface<B: BlockDevice>
where
  [(); nearest_div_8(B::NUM_BLOCKS)]: , {
  stored: [Option<Box<dyn Metadata>>; MD_SPACE],
  owners: [u8; MD_SPACE],

  // TODO below is the number of blocks, just picked an arbitrary number for now
  free_map: BitArray<{ B::NUM_BLOCKS }>,

  // TODO decide whether or not to include a BlockDescriptor sort of thing with an offset?
  block_device: *mut B,
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub struct MetadataHandle(u32);

static mut GLOBAL_BLOCK_INTERFACE: GlobalBlockInterface<Driver> =
  GlobalBlockInterface::new(unsafe { &mut DRIVER });

pub fn global_block_interface() -> &'static mut GlobalBlockInterface<Driver> {
  unsafe { &mut GLOBAL_BLOCK_INTERFACE }
}

fn as_dyn(md: &Box<dyn Metadata>) -> &dyn Any { md }

impl<B: BlockDevice> GlobalBlockInterface<B>
where
  [(); nearest_div_8(B::NUM_BLOCKS)]: ,
{
  const fn new(block_device: *mut B) -> Self {
    // TODO load prior if it exists here
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
      owners: [NO_OWNER; MD_SPACE],
      free_map: BitArray::new(false),
      block_device,
    }
  }

  /// Allocates new metadata inside of the system
  pub fn new_metadata<M: Metadata>(&mut self, owner_id: u8) -> Result<MetadataHandle, ()> {
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
    self.owners[free_space] = owner_id;
    Ok(MetadataHandle(free_space as u32))
  }

  /// Gets a reference to an instance of Metadata
  pub fn md_ref<M: Metadata>(&mut self, MetadataHandle(i): MetadataHandle) -> Result<&M, ()> {
    let md = self.stored.get(i as usize).ok_or(())?;
    let md = md.as_ref().ok_or(())?;
    let md = as_dyn(&md).downcast_ref::<M>().ok_or(())?;
    Ok(md)
  }

  /// Returns the metadata for a given owner_id, used after rebooting the system.
  pub fn metadatas_for(&self, owner_id: u8) -> impl Iterator<Item = MetadataHandle> + '_ {
    self
      .owners
      .iter()
      .enumerate()
      .filter(move |(_, &v)| v == owner_id)
      .map(|(i, _)| MetadataHandle(i as u32))
  }

  /// Requests an additional block for a specific MetadataHandle. The allocated block is not
  /// guaranteed to be contiguous.
  pub fn req_block<M: Metadata>(
    &mut self,
    MetadataHandle(i): MetadataHandle,
    new_block: u32,
  ) -> Result<(), ()> {
    if self.free_map.get(new_block) {
      return Err(());
    }
    let i = i as usize;
    let md = self.stored.get(i).ok_or(())?;
    let md = md.as_ref().ok_or(())?;
    let mut old_owned = md.owned().iter();

    // let new_b = self.free_map.find_free().ok_or(())? as u32;
    let new_md = as_dyn(&md).downcast_ref::<M>().unwrap().insert(new_block)?;

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
      Some(&b) if b == new_block => {},
      // inserted wrong block
      Some(_) => return Err(()),
      // no block seems to have been inserted
      None => return Err(()),
    };

    if old_owned.next().is_some() {
      // Some block was removed and not kept in the new metadata
      return Err(());
    }
    // perform updates after all checks
    self.free_map.set(new_block as usize);
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
