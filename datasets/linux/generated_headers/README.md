# Generated Headers

Header files generated during build. These are config and target arch
dependent. See [config](config) for the configuration options used to generate
these headers.

The steps to produce these headers is, from a host Linux machine:

```sh
# Get the sources.
$ wget https://github.com/torvalds/linux/archive/v4.19.tar.gz
$ extract v4.18.tar.gz
$ cd linux-4.19

# "Configure" and build
$ cp /boot/config-* .config
$ make menuconfig
$ make
$ find . -type d -name 'generated'

# Copy generated directories to this source tree
export DST="$HOME/phd/datasets/linux/generated_headers"
mkdir -p "$DST/arch/x86/include/generated" \
    "$DST"/tools/testing/radix-tree/generated \
    "$DST"/include/generated
rsync -avh arch/x86/include/generated/ "$DST/arch/x86/include/generated/"
rsync -avh tools/testing/radix-tree/generated/ "$DST"/tools/testing/radix-tree/generated/
rsync -avh include/generated/ "$DST"/include/generated/
cp include/linux/kconfig.h $DST/include/linux
cp include/linux/compiler_types.h $DST/include/linux
```
