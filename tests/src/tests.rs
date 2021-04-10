use crate::{block_interface::*, fs, linux_files};

macro_rules! fs_test {
  ($test_name: ident, |$fs_name: ident| $contents: block) => {
    #[test]
    fn $test_name() {
      static mut GBI: GlobalBlockInterface<linux_files::Driver> =
        GlobalBlockInterface::new(linux_files::Driver::new(stringify!($test_name)));
      let mut $fs_name = unsafe {
        GBI.try_init().expect("Failed to init interface");
        fs::FileSystem::new(&mut GBI)
      };
      $contents($fs_name);
      unsafe {
        GBI.block_device.clean();
      }
    }
  };
}

fs_test!(do_init, |gfs| {
  gfs
    .root_dir(fs::FileMode::R)
    .expect("failed to open root dir");
});

fs_test!(add_files_to_root_dir, |gfs| {
  let root_dir_fd = gfs
    .root_dir(fs::FileMode::R)
    .expect("failed to open root dir");
  for i in 0..50 {
    let fh = gfs
      .open(root_dir_fd, &[&i.to_string()], fs::FileMode::RW)
      .expect(&format!("Failed to create {}", i));
    let foo_bar = b"foo_bar";
    gfs.write(fh, foo_bar.as_slice()).expect("Failed to write");
  }
});

fs_test!(read_files_from_root_dir, |gfs| {
  let root_dir_fd = gfs
    .root_dir(fs::FileMode::R)
    .expect("failed to open root dir");
  for i in 0..50 {
    let fh = gfs
      .open(root_dir_fd, &[&i.to_string()], fs::FileMode::RW)
      .expect(&format!("Failed to create {}", i));
    let foo_bar = b"foo_bar";
    gfs.write(fh, foo_bar.as_slice()).expect("Failed to write");
    gfs
      .seek(fh, fs::SeekFrom::Start(0))
      .expect("Failed to seek");
    let mut buf = [0; 7];
    gfs.read(fh, &mut buf[..]).expect("Failed to read");
    assert_eq!(&buf, foo_bar);
    let static_str = b"this is a static string";
    gfs
      .write(fh, static_str.as_slice())
      .expect("Failed to write");
    let mut buf = [0; 23];
    gfs
      .seek(fh, fs::SeekFrom::Current(-(buf.len() as i32)))
      .expect("Failed to seek");
    gfs.read(fh, &mut buf).expect("Failed to read");
    assert_eq!(&buf, static_str);
  }
});

fs_test!(open_and_close_a_lot, |gfs| {
  let root_dir_fd = gfs
    .root_dir(fs::FileMode::R)
    .expect("failed to open root dir");
  for i in 0..50 {
    let _ = gfs
      .open(root_dir_fd, &[&i.to_string()], fs::FileMode::RW)
      .expect(&format!("Failed to create {}", i));
    let foo_bar = b"foo_bar";
    let () = gfs
      .unlink(root_dir_fd, &[&i.to_string()])
      .expect(&format!("Failed to unlink {}", i));
  }
  let dir = gfs
    .as_directory(root_dir_fd)
    .expect("Failed to get directory");

  assert_eq!(dir.entries().count(), 2);
  assert_eq!(dir.num_added, 0);
});

fs_test!(nested_directories, |gfs| {
  let root_dir_fd = gfs
    .root_dir(fs::FileMode::R)
    .expect("failed to open root dir");
  let dir2_fd = gfs.mkdir(root_dir_fd, "example_dir").expect("Failed to make directory");

  let dir = gfs
    .as_directory(root_dir_fd)
    .expect("Failed to get directory");
  let dir2 = gfs
    .as_directory(dir2_fd)
    .expect("Failed to get directory");

  assert_eq!(dir.entries().count(), 3);
  assert_eq!(dir.num_added, 1);

  assert_eq!(dir2.entries().count(), 2);
  assert_eq!(dir2.num_added, 0);

  let new_fd = gfs.open(dir2_fd, &["test.txt"], fs::FileMode::RW).expect("Failed to open file");
  gfs.write(new_fd, (b"Example text").as_slice()).unwrap();
  gfs.seek(new_fd, fs::SeekFrom::Start(0));
  let mut buf = [0; 12];
  gfs.read(new_fd, &mut buf).unwrap();
  assert_eq!("Example text".as_bytes(), &buf);

  // need to pull directory which is not updated.
  let dir2 = gfs
    .as_directory(dir2_fd)
    .expect("Failed to get directory");

  assert_eq!(dir2.entries().count(), 3);
  assert_eq!(dir2.num_added, 1);
});
