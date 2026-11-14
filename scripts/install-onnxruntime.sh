#!/bin/sh
# Downloads the ONNX Runtime shared library needed by vad/silero and installs
# it into a directory.
#
#   scripts/install-onnxruntime.sh [dir]      default: /usr/local/lib
#
# The archive comes from the official microsoft/onnxruntime GitHub release and
# is checked against a pinned SHA-256. github.com/yalue/onnxruntime_go v1.38.0
# is built with the 1.31 C API headers; ONNX Runtime keeps older API versions
# available, and the vad/silero tests pass with this newer library.
set -eu

VERSION=1.32.0
DEST="${1:-/usr/local/lib}"

case "$(uname -s)-$(uname -m)" in
Linux-x86_64)
	asset="onnxruntime-linux-x64-$VERSION"
	sum=bb169b7e374731730d2ae406fb39de7be4b315f413a9132efe896261c31bb503
	lib="libonnxruntime.so.$VERSION"
	link=libonnxruntime.so
	;;
Linux-aarch64 | Linux-arm64)
	asset="onnxruntime-linux-aarch64-$VERSION"
	sum=81a46251dd90da4122f4b9ceb4433f836fbc79ee1eddffca7c09033af81a1939
	lib="libonnxruntime.so.$VERSION"
	link=libonnxruntime.so
	;;
Darwin-arm64)
	asset="onnxruntime-osx-arm64-$VERSION"
	sum=8ddca23f532b9d4fbb279ba9b08dab6c0ac94fdc9aec2eb44a1db7049216c043
	lib="libonnxruntime.$VERSION.dylib"
	link=libonnxruntime.dylib
	;;
*)
	echo "install-onnxruntime: no pinned archive for $(uname -s)-$(uname -m);" >&2
	echo "install ONNX Runtime $VERSION yourself and set ONNXRUNTIME_LIB" >&2
	exit 1
	;;
esac

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

curl -fsSL -o "$tmp/ort.tgz" \
	"https://github.com/microsoft/onnxruntime/releases/download/v$VERSION/$asset.tgz"

if command -v sha256sum >/dev/null 2>&1; then
	got="$(sha256sum "$tmp/ort.tgz" | cut -d' ' -f1)"
else
	got="$(shasum -a 256 "$tmp/ort.tgz" | cut -d' ' -f1)"
fi
if [ "$got" != "$sum" ]; then
	echo "install-onnxruntime: checksum mismatch for $asset.tgz" >&2
	echo "  want $sum" >&2
	echo "  got  $got" >&2
	exit 1
fi

tar -xzf "$tmp/ort.tgz" -C "$tmp"
mkdir -p "$DEST"
cp "$tmp/$asset/lib/$lib" "$DEST/$lib"
ln -sf "$lib" "$DEST/$link"
echo "$DEST/$link"
