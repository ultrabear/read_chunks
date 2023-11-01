//! A crate that provides a [`read_chunks`][ReadExt::read_chunks] extension to types implementing [`std::io::Read`] (including unsized ones).
//!
//! # Motivation
//! Sometimes you may be reading a file to the end to do processing on it, but do not want the
//! entire file in memory. Sometimes [`bytes`] is the answer to this, but if you wish to
//! process larger chunks of data at once, maybe for SIMD, that cannot be used.
//!
//! Calling into [`read`] repeatedly to get a chunk until the end is tedious as it may return
//! significantly less bytes than you expected (slowing down bulk processing), or encounter a recoverable
//! error, and handling that yourself is a chore.
//!
//! A more correct implementation may be to use [`read_exact`] for that purpose, as it
//! guarantees the whole chunk. The problem with that is at the end of the file you will lose the
//! data that was read, as [`read_exact`] leaves the buffer contents unspecified at EOF.
//!
//! The method implemented in this crate addresses both problems, it guarantees the full buffer size
//! requested whenever it can, handles recoverable errors, and at the end of the Read stream will
//! simply return a final smaller buffer before returning [`None`] to signal the end of the stream was
//! detected. That is to say, you will always get the full buffer length until the last chunk where
//! you get a tail, similar to [`slice::chunks`][slice::chunks], but for a [`Read`].
//!
//! # Usage
//! Simply add `use read_chunks::ReadExt;` to your module and use the new [`read_chunks`][ReadExt::read_chunks] method that
//! should appear on any type implementing [`Read`].
//!
//! # Standard Library Inclusion
//! This crate was written because it is useful to me for hashing files incrementally with SIMD
//! optimized hashing functions like [blake3](https://crates.io/crates/blake3). This crate may
//! attempt to be added to the rust standard library if it is seen as generally useful and people
//! agree with the design. For this reason, the api may break in order to prototype
//! what could work best for the standard library.
//!
//! In particular, a `read_chunks_exact` api may be desirable that mirrors [`slice::chunks_exact`],
//! giving the remainder in a separate function, and asserting the length of the buffer in the main
//! iterator remains constant,
//!
//! It may also be put into question if the return type should be `&[u8]` or `&mut [u8]`. Currently
//! a `&mut [u8]` is returned because that is allowed by the implementation, but whether that makes
//! sense as an api is unknown at this time.
//!
//! [`bytes`]: Read::bytes
//! [`read`]: Read::read
//! [`read_exact`]: Read::read_exact

#![warn(clippy::pedantic)]
#![forbid(unsafe_code)]
#![warn(missing_docs, clippy::missing_docs_in_private_items)]

use std::io::{self, ErrorKind, Read};

/// Trait that extends any type implementing [`Read`] with [`ReadChunks`]
pub trait ReadExt: Read {
    /// Returns a lending iterator that yields chunks of size `n` until the end of the reader.
    ///
    /// This method will allocate n bytes *once*.
    fn read_chunks(&mut self, n: usize) -> ReadChunks<'_, Self> {
        ReadChunks {
            reader: self,
            // this *should* allocate once, please advise if not the case
            buffer: vec![0; n].into_boxed_slice(),
        }
    }
}

impl<T: Read + ?Sized> ReadExt for T {}

/// A lending iterator that allows reading chunks of `n` bytes at a time from a reader.
pub struct ReadChunks<'a, R: ?Sized + Read> {
    /// The backing reader
    reader: &'a mut R,
    /// Internal buffer that is filled on each [`ReadChunks::next_chunk`] call.
    /// It makes sense to use the `Cursor`/`buf` api if that becomes stable, as that can avoid
    /// zeroing the buffer initially.
    buffer: Box<[u8]>,
}

/// Tries to read to fill up the whole buffer, returning Ok(n) even if the function only managed to fill up n
/// bytes which was not the length of buf (this implies an EOF was reached)
fn try_read_exact<R: Read + ?Sized>(reader: &mut R, mut buf: &mut [u8]) -> io::Result<usize> {
    let mut read = 0;

    // based heavily on std::io::default_read_exact and default_read_to_end
    while !buf.is_empty() {
        match reader.read(buf) {
            Ok(0) => break,
            Ok(n) => {
                read += n;
                buf = &mut buf[n..];
            }
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            // if we want leftover data to still be accessible E should be (usize, io::Error) with
            // `read`. for now we make it just io::Error.
            Err(e) => return Err(e),
        }
    }

    Ok(read)
}

impl<R: Read + ?Sized> ReadChunks<'_, R> {
    /// Reads the next chunk of bytes from R with a size `n` specified by [`ReadExt::read_chunks`].
    ///
    /// This method is meant to be called repeatedly until None is returned, at which EOF is
    /// assumed to have occurred (when [`Read::read`] returns Ok(0)).
    ///
    ///
    ///
    /// # Errors
    /// If this function encounters [`ErrorKind::Interrupted`] it will continue to attempt to fill the buffer
    /// until a different error is encountered, the buffer is filled, or Read returns Ok(0).
    ///
    /// If a different read error occurs, this function will return the error, and the amount that
    /// was read into the internal buffer before the error is unspecified (another call to
    /// `next_chunk` will clobber this data). In the future, a way to extract the leftover buffer
    /// after an error may be added. This method was chosen as it mirrors what [`Read::read_exact`]
    /// does on an error.
    ///
    /// # Examples
    /// ```rust
    /// # use std::io::Read;
    /// # use read_chunks::ReadExt;
    /// let mut slice: &[u8] = &[0u8, 1, 2, 3];
    ///
    /// let mut it = slice.read_chunks(2);
    ///
    /// while let Some(chunk) = it.next_chunk() {
    ///     // unwrap the io error, real code should handle this
    ///     let chunk = chunk.unwrap();
    ///
    ///     assert!(chunk == &[0, 1] || chunk == &[2, 3]);
    /// }
    ///
    /// // The slice implementation of `Read` will empty the slice.
    /// // It is notable that read_chunks did not take ownership of 
    /// // slice however, only an exclusive borrow.
    /// assert!(slice.is_empty());
    /// ```
    ///
    pub fn next_chunk(&mut self) -> Option<io::Result<&mut [u8]>> {
        let res = try_read_exact(&mut self.reader, &mut self.buffer);

        match res {
            Ok(0) => None,
            Ok(n) => Some(Ok(&mut self.buffer[..n])),
            Err(e) => Some(Err(e)),
        }
    }
}
