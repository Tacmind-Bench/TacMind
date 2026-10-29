# Optional upstream assets

[中文](optional-assets.zh-CN.md) · [Policy guide](../README.md#policies)

TacMind policy training and evaluation use TacMind assets. ALOHA hardware CAD and
LIBERO simulation assets are optional and are not bundled with the policy sources.

To run the upstream OpenPI LIBERO example, download and extract its matching assets first.
Run from the TacMind repository root:

```bash
mkdir -p data/downloads policy/openpi/third_party/libero/libero/libero/assets
curl -fL https://github.com/Lifelong-Robot-Learning/LIBERO/archive/f78abd68ee283de9f9be3c8f7e2a9ad60246e95c.tar.gz \
  -o data/downloads/libero-assets.tar.gz
tar -xzf data/downloads/libero-assets.tar.gz --strip-components=4 \
  -C policy/openpi/third_party/libero/libero/libero/assets \
  LIBERO-f78abd68ee283de9f9be3c8f7e2a9ad60246e95c/libero/libero/assets
```

Then follow the [OpenPI LIBERO example](openpi/examples/libero/README.md).
For ALOHA hardware fabrication, use the [matching upstream CAD files](https://github.com/Physical-Intelligence/aloha/tree/d1dc83afd89ded4379851257fe5d85632d31d5ec/aloha2).
