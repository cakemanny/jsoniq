use std::env;
use std::io;
use std::io::Write;
use std::process;

use jsoniq::parse;

fn print_usage(mut w: impl Write) -> Result<(), Box<dyn std::error::Error>> {
    writeln!(w, "jsoniq")?;
    writeln!(w)?;
    writeln!(w, "usage: jsoniq [options] <jsoniq expression>")?;
    writeln!(w)?;
    Ok(())
}

fn main() {

    let args: Vec<_> = env::args().collect();

    if args.len() < 2 {
        print_usage(std::io::stderr()).unwrap();
        process::exit(1);
    }

    if args[1] == "-r" {
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
        }
    }

    if let Err(e) = parse_and_eval(&args[1]) {
        eprintln!("error: {}", e);
        process::exit(1);
    }
}

fn parse_and_eval(program: &str) -> anyhow::Result<()> {
    let (_, expr) = parse::parse_flwor(program)
        .map_err(|e| anyhow::Error::new(e.to_owned()))?;

    jsoniq::eval_query(&expr)?.for_each(|value| println!("{:?}", value));

    Ok(())
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
    parse_and_eval(&buffer)?;

    Ok(LoopState::Continue)
}
