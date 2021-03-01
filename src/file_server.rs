
static mut FILE_DESCS: [FileDescriptor; 256] = [FileDescriptor; 256];

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct FileHandle(u32);


pub fn open(name: &str) -> Result<FileHandle> {
}
