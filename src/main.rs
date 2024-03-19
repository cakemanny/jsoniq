use std::io;

use jsoniq::parse;
use jsoniq::run_example;

fn main() {
    loop {
        match rep() {
            Ok(LoopState::Continue) => {}
            Ok(LoopState::End) => {
                return;
            }
            Err(e) => {
                println!("error: {}", e);
            }
        }
        if false {
            run_example();
        }
    }
}

enum LoopState {
    Continue,
    End,
}

/// REPL without the L
fn rep() -> anyhow::Result<LoopState> {
    let mut buffer = String::new();
    let bytes_read = io::stdin().read_line(&mut buffer)?;
    if bytes_read == 0 {
        return Ok(LoopState::End);
    }

    let (_, expr) = parse::parse_flwor(&buffer)
        .map_err(|e| anyhow::Error::new(e.to_owned()))?;

    jsoniq::eval_query(&expr)?.for_each(|value| println!("{:?}", value));

    Ok(LoopState::Continue)
}
