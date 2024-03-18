use std::io;

use jsoniq::parse;
use jsoniq::run_example;

fn main() {
    match rep() {
        Ok(()) => {}
        Err(e) => {
            println!("error: {}", e);
            return;
        }
    }

    if false {
        run_example();
    }
}

fn rep() -> anyhow::Result<()> {
    let mut buffer = String::new();
    io::stdin().read_line(&mut buffer)?;

    // I was unable to work out any other way
    let buf_contents: &'static str = Box::leak(buffer.into_boxed_str());
    match parse::parse_flwor(buf_contents) {
        Ok((_, expr)) => {
            jsoniq::eval_query(&expr)?
                .for_each(|value| println!("{:?}", value));
            Ok(())
        }
        Err(e) => Err(anyhow::Error::new(e)),
    }
}
