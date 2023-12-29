import os
import io
import urllib.parse
import contextlib
import pyarrow
from webdataset import reraise_exception
from webdataset.shardcache import guess_shard


def hdfs_download(uri, **kwargs):
    if uri.hostname is not None:
        kwargs.update(host=uri.hostname)
    if uri.port is not None:
        kwargs.update(post=uri.post)
    with contextlib.closing(pyarrow.hdfs.connect(**kwargs)) as hdfs:
        with hdfs.open(uri.path) as stream:
            return stream.read()


def file_download(uri, *args, **kwargs):
    with open(uri.path, mode="rb") as input:
        return input.read()


download_schemes = dict(file=file_download, hdfs=hdfs_download)


def cached_download_shard(url, cache_dir, *args, **kwargs):
    file_name = guess_shard(url)
    cache_path = os.path.join(cache_dir, file_name)
    if os.path.exists(cache_path):
        return cache_path
    uri = urllib.parse.urlparse(url, scheme="file", allow_fragments=False)
    if uri.scheme not in download_schemes:
        raise ValueError(f"{uri.scheme}: unknown source")
    data = download_schemes[uri.scheme](uri, *args, **kwargs)
    temp_cache_path = os.path.join(
        cache_dir, file_name + ".~" + str(os.getpid()) + "~" "_"
    )
    with open(temp_cache_path, "wb") as cache_file:
        cache_file.write(data)
    os.rename(temp_cache_path, cache_path)
    seekable_stream = io.BytesIO(data)
    return seekable_stream


def download_shard(url, *args, **kwargs):
    uri = urllib.parse.urlparse(url, scheme="file", allow_fragments=False)
    if uri.scheme not in download_schemes:
        raise ValueError(f"{uri.scheme}: unknown source")
    if uri.scheme in ("file", "hdfs"):
        return url
    data = download_schemes[uri.scheme](uri, *args, **kwargs)
    seekable_stream = io.BytesIO(data)
    return seekable_stream


def download_shards(shards, cache_dir=None, handler=reraise_exception, **kwargs):
    """Modification of webdataset.url_opener for pyarrow.parquet requirements"""
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
    for shard in shards:
        assert isinstance(shard, dict)
        assert "url" in shard
        try:
            if cache_dir is None:
                data = download_shard(shard["url"], **kwargs)
            else:
                data = cached_download_shard(shard["url"], cache_dir, **kwargs)

            shard.update(data=data)
            yield shard
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break
