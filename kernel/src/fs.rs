use crate::{
  bit_array::{nearest_div_8, BitArray},
  block_interface::{
    AllMetadata, BlockDevice, GlobalBlockInterface, Metadata, MetadataHandle, Owner,
    FIRST_FREE_BLOCK, OWN_BLOCKS,
  },
  default_ser_impl,
};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FileDescriptor(u32);

#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FileMode {
  R,
  W,
  RW,
}

impl Default for FileMode {
  fn default() -> Self { Self::R }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub struct FileDescEntry {
  inode: u32,
  offset: u32,
  open_refs: u16,
  mode: FileMode,
}

const NUM_ENTRIES: usize = 64;
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Directory {
  parent_inode: u32,
  own_inode: u32,
  num_added: u8,
  name_to_inode_map: [([u8; 48], u32); NUM_ENTRIES],
}

impl Directory {
  fn new(own_inode: u32, parent_inode: u32) -> Self {
    Self {
      parent_inode,
      own_inode,
      num_added: 0,
      // The last byte in this array is the length, which is super hacky.
      name_to_inode_map: [([0; 48], 0); NUM_ENTRIES],
    }
  }
  /// Returns all the byte array entries of this directory
  #[inline]
  pub(crate) fn entries(&self) -> impl Iterator<Item = (&[u8], u32)> {
    core::iter::once((b".".as_slice(), self.own_inode))
      .chain(core::iter::once((b"..".as_slice(), self.parent_inode)))
      .chain(
        self
          .name_to_inode_map
          .iter()
          .take(self.num_added as usize)
          .map(|(name, inode)| (&name[..name[47] as usize], *inode)),
      )
  }
  /// Inserts an entry into this directory
  // TODO convert this into a result type
  pub fn insert(&mut self, name: &str, to: u32) {
    assert!(
      (self.num_added as usize) < NUM_ENTRIES,
      "Directory out of space"
    );
    let bytes = name.as_bytes();
    assert!(bytes.len() < 48, "Cannot store names this long currently.");
    let len = bytes.len();
    let idx = self.num_added as usize;
    self.name_to_inode_map[idx].0[..len].copy_from_slice(bytes);
    self.name_to_inode_map[idx].0[47] = len as u8;
    self.name_to_inode_map[idx].1 = to;
    self.num_added += 1;
  }
  /// Removes an entry from this directory
  pub fn remove(&mut self, entry_num: usize) {
    self
      .name_to_inode_map
      .swap(entry_num, (self.num_added - 1) as usize);
    self.num_added -= 1;
  }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct INode {
  refs: u16,
  kind: INodeKind,
  size: u32,
  // TODO think of a way to extend the size of data blocks here
  data_blocks: [u16; 8],
}

impl INode {
  const fn new(kind: INodeKind) -> Self {
    INode {
      refs: 1,
      kind,
      size: 0,
      data_blocks: [0; 8],
    }
  }
  /// Creates an inode from a slice of bytes
  #[inline]
  fn from_slice(s: &[u8]) -> Self {
    assert_eq!(s.len(), core::mem::size_of::<Self>());
    let mut data_blocks = [0u16; 8];
    for i in 0..8 {
      data_blocks[i] = u16::from_ne_bytes([s[7 + i * 2], s[8 + i * 2]]);
    }
    Self {
      refs: u16::from_ne_bytes([s[0], s[1]]),
      kind: INodeKind::from(s[2]),
      size: u32::from_ne_bytes([s[3], s[4], s[5], s[6]]),
      data_blocks,
    }
  }
  fn to_slice(&self, dst: &mut [u8]) {
    assert_eq!(dst.len(), core::mem::size_of::<Self>());
    let mut data_blocks = [0u16; 8];
    dst[..2].copy_from_slice(&self.refs.to_ne_bytes());
    dst[2] = self.kind as u8;
    dst[3..7].copy_from_slice(&self.size.to_ne_bytes());
    for i in 0..8 {
      let [l, r] = self.data_blocks[i].to_ne_bytes();
      dst[7 + i * 2] = l;
      dst[8 + i * 2] = r;
    }
  }
}

#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum INodeKind {
  File = 0,
  Directory = 1,
}

impl From<u8> for INodeKind {
  fn from(v: u8) -> Self {
    match v {
      0 => INodeKind::File,
      1 => INodeKind::Directory,
      v => panic!("Unknown INodeKind {}", v),
    }
  }
}

const MAGIC_NUMBER: u32 = 0x101_0_F_0ff;

/// The superblock is where the block alloc maps are stored
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Superblock {
  magic_number: u32,
  owned: Option<[u32; 1]>,
}

impl Metadata for Superblock {
  fn new() -> Self {
    Superblock {
      magic_number: MAGIC_NUMBER,
      owned: None,
    }
  }
  type Iter<'a> = core::iter::Copied<core::slice::Iter<'a, u32>>;
  fn owned(&self) -> Self::Iter<'_> {
    match self.owned {
      None => [].iter(),
      Some(ref v) => v.iter(),
    }
    .copied()
  }
  fn insert(&self, b: u32) -> Result<Self, ()> {
    if self.owned.is_some() {
      return Err(());
    }
    Ok(Self {
      owned: Some([b]),
      ..*self
    })
  }
  fn remove(&self, b: u32) -> Result<Self, ()> {
    if self.owned.is_none() {
      return Err(());
    }
    Ok(Self {
      owned: None,
      ..*self
    })
  }
  const LEN: usize = core::mem::size_of::<Self>();
  default_ser_impl!();
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RangeMetadata {
  start: u32,
  count: u32,
}

impl Metadata for RangeMetadata {
  fn new() -> Self { RangeMetadata { start: 0, count: 0 } }
  type Iter<'a> = core::ops::Range<u32>;
  fn owned(&self) -> Self::Iter<'_> { self.start..self.start + self.count }
  const LEN: usize = core::mem::size_of::<Self>();
  fn insert(&self, b: u32) -> Result<Self, ()> {
    if self.count == 0 {
      return Ok(Self { start: b, count: 1 });
    }
    // Handle an extra block after the end of this one
    if b == self.start + self.count {
      return Ok(Self {
        start: self.start,
        count: self.count + 1,
      });
    } else if b == self.start.checked_sub(1).ok_or(())? {
      return Ok(Self {
        start: self.start - 1,
        count: self.count + 1,
      });
    }
    Err(())
  }

  fn remove(&self, b: u32) -> Result<Self, ()> {
    todo!();
  }

  default_ser_impl!();
}

/// The number of inodes this file system supports.
pub const NUM_INODE: usize = 1024;
/// The number of data blocks in this system.
pub const NUM_DATA: usize = 2048;
// TODO for some reason I can't just plug it in below so I need to hard code this

/// A singleton file system type.
pub struct FileSystem<B: 'static + BlockDevice>
where
  B: BlockDevice,
  [(); nearest_div_8(B::NUM_BLOCKS)]: ,
  [(); B::BLOCK_SIZE]: , {
  superblock: MetadataHandle,
  open_files: [FileDescEntry; 256],
  inode_md: MetadataHandle,
  data_md: MetadataHandle,
  inode_alloc_map: BitArray<1024>,
  data_alloc_map: BitArray<2048>,
  gbi: &'static mut GlobalBlockInterface<B>,
}

impl<B> FileSystem<B>
where
  B: BlockDevice,
  [(); nearest_div_8(B::NUM_BLOCKS)]: ,
  [(); B::BLOCK_SIZE]: ,
  [(); 2 * B::BLOCK_SIZE]: ,
  [(); OWN_BLOCKS * B::BLOCK_SIZE]: ,
{
  pub fn new(gbi: &'static mut GlobalBlockInterface<B>) -> Self {
    let sb_mh = gbi
      .metadatas_for(Owner::LibFS)
      .find_map(|(mh, amd)| matches!(amd, AllMetadata::Superblock(_)).then(|| mh));
    if let Some(sb_mh) = sb_mh {
      // File system previously existed, reinitialize
      let mut mhs = gbi
        .metadatas_for(Owner::LibFS)
        .filter(|(mh, amd)| matches!(amd, AllMetadata::RangeMetadata(_)))
        .map(|v| v.0);
      let inode_mh = mhs.next().expect("Got inode metadata handle");
      let data_mh = mhs.next().expect("Got inode metadata handle");
      assert!(mhs.next().is_none());
      drop(mhs);
      let mut out = Self {
        superblock: sb_mh,
        inode_md: inode_mh,
        data_md: data_mh,
        inode_alloc_map: BitArray::new(false),
        data_alloc_map: BitArray::new(false),
        open_files: [FileDescEntry::default(); 256],
        gbi,
      };
      out.load_allocs().expect("Failed to load alloc map");
      out
    } else {
      Self::mk(gbi)
    }
  }
  /// Makes a new instance of this file system on this block interface
  pub fn mk(gbi: &'static mut GlobalBlockInterface<B>) -> Self {
    let sb_mh = gbi
      .new_metadata::<Superblock>(Owner::LibFS)
      .expect("Failed to mk sb md");
    gbi
      .req_block(sb_mh, FIRST_FREE_BLOCK)
      .expect("Failed to initialize Super block");
    let inode_mh = gbi
      .new_metadata::<RangeMetadata>(Owner::LibFS)
      .expect("Failed to mk inode md");
    let mut curr = FIRST_FREE_BLOCK + 1;
    let num_inode_blocks = Self::num_inode_blocks() as u32;
    for i in curr..curr + num_inode_blocks {
      gbi
        .req_block(inode_mh, i)
        .expect("Failed to add block to inode");
    }
    curr += num_inode_blocks;
    let data_mh = gbi
      .new_metadata::<RangeMetadata>(Owner::LibFS)
      .expect("Failed to mk data md");
    for i in curr..curr + NUM_DATA as u32 {
      gbi
        .req_block(data_mh, i)
        .expect("Failed to add block to data");
    }
    gbi.persist();
    let mut out = Self {
      superblock: sb_mh,
      inode_md: inode_mh,
      data_md: data_mh,
      inode_alloc_map: BitArray::new(false),
      data_alloc_map: BitArray::new(false),
      open_files: [FileDescEntry::default(); 256],
      gbi,
    };
    // make root directory at inode 0.
    let mut root_dir_inode = out
      .alloc_inode(&INode::new(INodeKind::Directory))
      .expect("Failed to allocate inode for root directory");
    assert_eq!(root_dir_inode, 0, "Unexpected inode for root dir");
    let root_dir = Directory::new(0, 0);
    out
  }
  #[inline]
  const fn num_inode_blocks() -> usize {
    let mut num_blocks_for_inodes = (NUM_INODE * core::mem::size_of::<INode>() / B::BLOCK_SIZE);
    if NUM_INODE * core::mem::size_of::<INode>() % B::BLOCK_SIZE != 0 {
      num_blocks_for_inodes += 1;
    }
    num_blocks_for_inodes
  }
  /// Returns the position of an inode and whether it wraps around to a second block
  const fn inode_block_and_offset_and_wraps(i: usize) -> (usize, usize, bool) {
    let block = (i * core::mem::size_of::<INode>()) / B::BLOCK_SIZE;
    let overlaps_end = block == (((i + 1) * core::mem::size_of::<INode>()) / B::BLOCK_SIZE);
    let offset = (i * core::mem::size_of::<INode>()) % B::BLOCK_SIZE;
    (block, offset, overlaps_end)
  }
  fn load_inode(&self, i: usize) -> Result<INode, ()> {
    assert!(
      self.inode_alloc_map.get(i),
      "Unallocated inode is being loaded"
    );
    let (block, offset, overlaps_end) = Self::inode_block_and_offset_and_wraps(i);
    if !overlaps_end {
      let mut buf = [0; B::BLOCK_SIZE];
      self.gbi.read(self.inode_md, block, &mut buf)?;
      Ok(INode::from_slice(
        &buf[offset..offset + core::mem::size_of::<INode>()],
      ))
    } else {
      let mut buf = [0u8; 2 * B::BLOCK_SIZE];
      self
        .gbi
        .read(self.inode_md, block, &mut buf[..B::BLOCK_SIZE])?;
      self
        .gbi
        .read(self.inode_md, block + 1, &mut buf[B::BLOCK_SIZE..])?;
      Ok(INode::from_slice(
        &buf[offset..offset + core::mem::size_of::<INode>()],
      ))
    }
  }
  fn save_inode(&self, inode: &INode, i: usize) -> Result<(), ()> {
    assert!(
      self.inode_alloc_map.get(i as usize),
      "Unallocated inode is being saved"
    );
    let (block, offset, overlaps_end) = Self::inode_block_and_offset_and_wraps(i);
    if !overlaps_end {
      let mut buf = [0u8; B::BLOCK_SIZE];
      inode.to_slice(&mut buf[offset..offset + core::mem::size_of::<INode>()]);
      self.gbi.write(self.inode_md, block, &buf)?;
    } else {
      let mut buf = [0u8; 2 * B::BLOCK_SIZE];
      inode.to_slice(&mut buf[offset..offset + core::mem::size_of::<INode>()]);
      self
        .gbi
        .write(self.inode_md, block, &buf[..B::BLOCK_SIZE])?;
      self
        .gbi
        .write(self.inode_md, block + 1, &buf[B::BLOCK_SIZE..])?;
    }
    Ok(())
  }
  /// Given an inode, saves it and returns the inode number it was given
  fn alloc_inode(&mut self, inode: &INode) -> Result<u32, ()> {
    let inode_num = self.inode_alloc_map.find_free().ok_or(())? as u32;
    self.save_inode(inode, inode_num as usize);
    self.inode_alloc_map.set(inode_num as usize);
    self.persist_allocs()?;
    Ok(inode_num)
  }
  /// Frees an inode from the inode alloc map
  fn free_inode(&mut self, i: usize) -> Result<(), ()> {
    assert!(
      self.inode_alloc_map.get(i),
      "Trying to deallocate unallocated inode {}",
      i
    );
    self.inode_alloc_map.unset(i);
    self.persist_allocs()?;
    Ok(())
  }
  /// Makes a directory at the given path in the current directory.
  pub fn mkdir(&mut self, curr_dir: &Directory, path: &str) -> Result<(), ()> {
    todo!();
  }
  /// Opens a given path relative to dir.
  pub fn open(
    &self,
    curr_dir: &Directory,
    path: &[&str],
    mode: FileMode,
  ) -> Result<FileDescriptor, ()> {
    // let mut curr_inode: u32 = self.root;
    // for segment in path {}
    todo!()
  }
  pub fn close(&mut self, FileDescriptor(i): FileDescriptor) -> Result<(), ()> {
    let i = i as usize;
    let open_file = &mut self.open_files[i];
    if open_file.open_refs == 0 {
      return Err(());
    }
    open_file.open_refs -= 1;
    // If inode is no longer referenced here, need to delete it
    todo!();
    return Ok(());
  }
  /// Saves the allocation maps to disk
  fn persist_allocs(&self) -> Result<(), ()> {
    let mut buf = [0u8; { (NUM_INODE + NUM_DATA) / 8 }];
    assert!(buf.len() < B::BLOCK_SIZE);
    buf[..self.inode_alloc_map.items.len()].copy_from_slice(&self.inode_alloc_map.items);
    buf[self.inode_alloc_map.items.len()..].copy_from_slice(&self.data_alloc_map.items);
    self.gbi.write(self.superblock, 0, &buf)?;
    Ok(())
  }
  /// Loads the allocation maps from disk
  fn load_allocs(&mut self) -> Result<(), ()> {
    let mut buf = [0u8; { (NUM_INODE + NUM_DATA) / 8 }];
    self.gbi.read(self.superblock, 0, &mut buf)?;
    let len = self.inode_alloc_map.items.len();
    self.inode_alloc_map.items.copy_from_slice(&buf[..len]);
    self.data_alloc_map.items.copy_from_slice(&buf[len..]);
    Ok(())
  }

  /// Writes a byte array to an inode
  fn write_to_inode(&mut self, inode: &mut INode, data: &[u8], offset: u32) -> Result<(), ()> {
    let start_block = offset / (B::BLOCK_SIZE as u32);
    let end_byte = (offset + data.len() as u32);
    let end_block = (end_byte as usize + B::BLOCK_SIZE - 1) / (B::BLOCK_SIZE);
    if end_block > 8 {
      // Not enough space in these files to write that much.
      // TODO lift this when I figure out a way to allocate more space
      return Err(());
    }
    let curr_blocks = (inode.size as usize + B::BLOCK_SIZE - 1) / B::BLOCK_SIZE;
    // TODO check the number of free data blocks before and abort early if so;
    if curr_blocks < end_block {
      for i in curr_blocks + 1..end_block {
        let free_db = self.data_alloc_map.find_free().ok_or(())?;
        inode.data_blocks[i] = free_db as u16;
        self.data_alloc_map.set(free_db);
      }
    }
    inode.size = inode.size.max(end_byte);
    let mut written = 0;
    let mut buf = [0; B::BLOCK_SIZE];
    for i in start_block..end_block as u32 {
      let db = inode.data_blocks[i as usize] as usize;
      assert_eq!(self.gbi.read(self.data_md, db, &mut buf)?, B::BLOCK_SIZE);
      let write_buf = &mut buf[(written + offset as usize) % B::BLOCK_SIZE..];
      write_buf.copy_from_slice(&data[written..write_buf.len()]);
      written += self.gbi.write(self.data_md, db, &buf)?;
    }
    assert_eq!(written, data.len());
    Ok(())
  }
  /// Writes a byte array to an inode
  fn read_from_inode(&mut self, inode: &INode, dst: &mut [u8], offset: u32) -> Result<(), ()> {
    let start_block = offset / (B::BLOCK_SIZE as u32);
    let end_byte = (offset + dst.len() as u32);
    let end_block = (end_byte as usize + B::BLOCK_SIZE - 1) / (B::BLOCK_SIZE);
    if end_block > 8 {
      // Reading past the end of the file
      return Err(());
    }
    let curr_blocks = (inode.size as usize + B::BLOCK_SIZE - 1) / B::BLOCK_SIZE;
    if curr_blocks < end_block {
      return Err(());
    }
    let mut read = 0;
    let mut buf = [0; B::BLOCK_SIZE];
    for i in start_block..end_block.min(curr_blocks) as u32 {
      let db = inode.data_blocks[i as usize] as usize;
      assert_eq!(self.gbi.read(self.data_md, db, &mut buf)?, B::BLOCK_SIZE);
      let read_buf = &buf[(read + offset as usize) % B::BLOCK_SIZE..];
      dst[read..read_buf.len()].copy_from_slice(read_buf);
      read += read_buf.len();
    }
    assert_eq!(read, dst.len());
    Ok(())
  }
}
