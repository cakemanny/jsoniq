
A small implementation of my favourite parts of JSONiq


https://www.jsoniq.org/

https://en.wikipedia.org/wiki/JSONiq


EXAMPLE:

After building with `cargo build --release`

```shell
$ ./target/release/jsoniq 'for $x in (1,2,3,4,5), $y in (1,2,3,4,5) where $x lt 3 return [$x, $y]'
[1.0,1.0]
[1.0,2.0]
[1.0,3.0]
[1.0,4.0]
[1.0,5.0]
[2.0,1.0]
[2.0,2.0]
[2.0,3.0]
[2.0,4.0]
[2.0,5.0]
```


Why JSONiq?

I think `jq` is a really great tool for doing some quick transformations on
some line-oriented json. But as soon as it's necessary to do something a
little more complicated it, the terseness becomes a barrier.
JSONiq with its roots in query languages feels a much better fit for
filtering, joining and other slightly more advanced tasks.



Note, this is just in its infancy - just like my rust knowledge,
so no judging at this point :)
