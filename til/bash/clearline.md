# Clear the current line of output

The carriage return `\r` escape code allows you to overwrite the
current line of output, however it will not *overwrite* the existing
content, so if the next string is shorter then there will be an
overlap. To erase the contents of the line, use the code
`\033[K`. Example:

```sh
echo -n this is a long string; echo -re '\r\033[Kshort string'
```

Source: http://stackoverflow.com/a/6774395
