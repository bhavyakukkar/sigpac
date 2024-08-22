#![no_std]

extern crate alloc;
use alloc::{
    boxed::Box,
    vec::{IntoIter, Vec},
};

#[derive(Debug, PartialEq)]
pub struct ParseErr<E>(Vec<E>);

impl<E> IntoIterator for ParseErr<E> {
    type Item = E;
    type IntoIter = IntoIter<E>;
    fn into_iter(self) -> IntoIter<E> {
        self.0.into_iter()
    }
}

type ParseResult<S, P, E> = Result<(S, P), Option<ParseErr<E>>>;
type ParseFn<'a, S, P, E> = Box<dyn FnOnce(S) -> ParseResult<S, P, E> + 'a>;

/// A one-time-use parser that can parse a value of type `P` from a stream of tokens of type `T`,
/// propagating errors of type `E`.
///
/// The parse-function, which can be invoked by passing an input stream of tokens to [`Parser::run`]
/// can return three different things:
///
/// 1. `Ok((remaining-input, parsed-value))` --> Value successfully parsed from input
/// 2. `Err(None)` --> Value pattern not found in input
/// 3. `Err(Some(parse-error))` --> Value pattern partially present but incomplete due to an error
///
pub struct Parser<'a, S, P, E>(ParseFn<'a, S, P, E>)
where
    S: IntoIterator;

impl<'a, S, P, E> Parser<'a, S, P, E>
where
    S: IntoIterator + FromIterator<S::Item>,
{
    /// Create a new parser
    pub fn new<F>(run_parser: F) -> Self
    where
        F: Into<ParseFn<'a, S, P, E>>,
    {
        Parser(run_parser.into())
    }

    /// Run the parser by providing the stream of tokens
    pub fn run<I>(self, input: I) -> ParseResult<S, P, E>
    where
        I: Into<S>,
    {
        self.0(input.into())
    }

    /// Create a new parser that ignores the input and returns the provided parsed value
    pub fn pure<V>(parsed_value: V) -> Self
    where
        V: Into<P> + 'a,
        P: 'a,
    {
        Parser(Box::new(move |input| Ok((input, parsed_value.into()))))
    }
}

/// Create a new parser that attempts to parse the provided token from the input
pub fn find_one<'a, S, T, E>(t: T) -> Parser<'a, S, S::Item, E>
where
    S: IntoIterator + FromIterator<S::Item>,
    T: Into<S::Item> + 'a,
    S::Item: PartialEq + 'a,
{
    Parser(Box::new(move |input| {
        let mut input = input.into_iter();
        let t = t.into();
        match input.next() {
            Some(x) if x == t => Ok((S::from_iter(input), t)),
            _ => Err(None),
        }
    }))
}

/// Create a new parser that attempts to parse the provided stream of tokens from the input
pub fn find<'a, S, E, LT>(s: S) -> Parser<'a, S, LT, E>
where
    S: IntoIterator + FromIterator<S::Item> + 'a,
    E: 'a,
    S::Item: PartialEq + 'a,
    LT: IntoIterator<Item = S::Item> + FromIterator<S::Item> + 'a,
{
    s.into_iter().map(find_one).collect::<Vec<_>>().into()
}

/// Create a new parser that attempts to parse the stream of tokens while they satisfy a predicate
pub fn find_predicate<'a, S, E, LT, F>(predicate: F) -> Parser<'a, S, LT, E>
where
    S: IntoIterator + FromIterator<S::Item> + Clone,
    LT: IntoIterator<Item = S::Item> + FromIterator<S::Item> + 'a,
    F: Fn(&LT::Item) -> bool + Copy + 'a,
{
    Parser(Box::new(move |input| {
        Ok((
            S::from_iter(input.clone().into_iter().skip_while(predicate)),
            LT::from_iter(input.into_iter().take_while(predicate)),
        ))
    }))
}

/// Functions associated to parser of Iterators
impl<'a, S, E, LT> Parser<'a, S, LT, E>
where
    S: IntoIterator + FromIterator<S::Item> + 'a,
    E: 'a,
    LT: IntoIterator + FromIterator<LT::Item> + 'a,
{
    /// Create a new parser from an old parser that becomes unsuccessful on the event that the
    /// successfully parsed iterator (in the old parser) is empty
    pub fn not_null(self) -> Self {
        Parser(Box::new(|input| {
            let (input, list) = self.0(input)?;
            let mut list = list.into_iter();
            match list.next() {
                Some(first_elem) => {
                    Ok((input, LT::from_iter([first_elem].into_iter().chain(list))))
                }
                None => Err(None),
            }
        }))
    }
}

impl<'a, S, P, E> Parser<'a, S, P, E>
where
    S: IntoIterator + FromIterator<S::Item> + 'a,
    P: 'a,
    E: 'a,
{
    /// Create a new parser from the old parser that upon parsing, applies a function to the parsed
    /// value `A`
    pub fn map<B: 'a, F>(self, f: F) -> Parser<'a, S, B, E>
    where
        F: FnOnce(P) -> B + 'a,
    {
        Parser(Box::new(move |input| {
            let (input, a) = self.0(input)?;
            Ok((input, f(a)))
        }))
    }

    /// Create a new parser from an old parser and a parser that parses a closure, by applying the
    /// parsed closure on the value parsed by `self` on the remaining input
    pub fn sequence<N>(
        self,
        f: Parser<'a, S, Box<dyn FnOnce(P) -> N + 'a>, E>,
    ) -> Parser<'a, S, N, E>
    where
        N: 'a,
    {
        Parser(Box::new(|input| {
            let (input, f) = f.0(input)?;
            let (input, a) = self.0(input)?;
            Ok((input, f(a)))
        }))
    }

    /// Create a parser from two old parsers (that parse the same type) that attempts to parse input
    /// with `self` and then attempts to parse same input with `other` upon failure while combining
    /// any errors returned while parsing `self`
    pub fn assoc(self, rhs: Self) -> Self
    where
        S: Clone,
    {
        //TODO sum up errors in self & other as an when they are accessed
        //Err(None) + Err(None) = Err(None)
        //Err(None) + Err(Some(b)) = Err(Some(b))
        //Err(Some(a)) + Err(None) = Err(Some(a))
        //Err(Some(a)) + Err(Some(b)) = Err(Some(a + b))
        Parser(Box::new(|input| self.0(input.clone()).or(rhs.0(input))))
    }

    /// Create a new parser from two old parsers, that parses input sequentially using `self` then
    /// `rhs`, but returns only the parsed value of `rhs` if exists, and the sum of any errors if
    /// not
    pub fn seq_right<N>(self, rhs: Parser<'a, S, N, E>) -> Parser<'a, S, N, E>
    where
        N: 'a,
    {
        Parser(Box::new(|input| match self.0(input) {
            Ok((input, _)) => rhs.0(input),
            Err(e) => Err(e),
        }))
    }

    /// Create a new parser from two old parsers, that parses input sequentially using `self` then
    /// `rhs`, but returns only the parsed value of `self` if exists, and the sum of any erros if
    /// not
    pub fn seq_left<N>(self, rhs: Parser<'a, S, N, E>) -> Self
    where
        N: 'a,
    {
        Parser(Box::new(|input| match self.0(input) {
            Ok((input, a)) => rhs.0(input).map(|(input, _)| (input, a)),
            Err(e) => Err(e),
        }))
    }
}

/// Converts a list of parsers of a type into a parser of a list of that type that sequentially
/// attempts to parse each parser, chaining them along
impl<'a, S, P, LP, LT, E> From<LP> for Parser<'a, S, LT, E>
where
    S: IntoIterator + FromIterator<S::Item> + 'a,
    P: 'a,
    E: 'a,
    LP: IntoIterator<Item = Parser<'a, S, P, E>> + FromIterator<LP::Item>,
    LT: IntoIterator<Item = P> + FromIterator<LT::Item> + 'a,
{
    /// Converts a list of parsers of a type into a parser of a list of that type that sequentially
    /// attempts to parse each parser, chaining them along
    fn from(value: LP) -> Self {
        let mut it = value.into_iter();
        match it.next() {
            Some(parser) => <Self as From<LP>>::from(LP::from_iter(it)).sequence(parser.map(
                |inner| -> Box<dyn FnOnce(LT) -> LT> {
                    Box::new(|old_it| {
                        LT::from_iter(LT::from_iter([inner]).into_iter().chain(old_it))
                    })
                },
            )),
            None => Parser::pure(LT::from_iter([])),
        }
    }
}

mod operator_overloads {
    use crate::{ParseErr, Parser};

    use alloc::boxed::Box;
    use core::ops::{Add, Mul, Shl, Shr};

    /// `Parser<A> << Parser<A>` :: `Parser<A>.seq_left(Parser<A>)`
    impl<'a, S, P, N, E> Shl<Parser<'a, S, N, E>> for Parser<'a, S, P, E>
    where
        S: IntoIterator + FromIterator<S::Item> + 'a,
        P: 'a,
        N: 'a,
        E: 'a,
    {
        type Output = Self;
        fn shl(self, rhs: Parser<'a, S, N, E>) -> Self {
            self.seq_left(rhs)
        }
    }

    /// `Parser<A> >> Parser<A>` :: `Parser<A>.seq_right(Parser<A>)`
    impl<'a, S, P, N, E> Shr<Parser<'a, S, N, E>> for Parser<'a, S, P, E>
    where
        S: IntoIterator + FromIterator<S::Item> + 'a,
        P: 'a,
        N: 'a,
        E: 'a,
    {
        type Output = Parser<'a, S, N, E>;
        fn shr(self, rhs: Parser<'a, S, N, E>) -> Parser<'a, S, N, E> {
            self.seq_right(rhs)
        }
    }

    /// `Parser<A> + Parser<A>` :: `Parser<A>.assoc(Parser<A>)`
    impl<'a, S, P, E> Add for Parser<'a, S, P, E>
    where
        S: IntoIterator + FromIterator<S::Item> + Clone + 'a,
        P: 'a,
        E: 'a,
    {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            self.assoc(rhs)
        }
    }

    /// `Parser<A -> B> * Parser<A>` :: `Parser<A>.sequence(Parser<A -> B>)`
    impl<'a, S, P, E, N> Mul<Parser<'a, S, P, E>> for Parser<'a, S, Box<dyn FnOnce(P) -> N + 'a>, E>
    where
        S: IntoIterator + FromIterator<S::Item> + 'a,
        P: 'a,
        E: 'a,
        N: 'a,
    {
        type Output = Parser<'a, S, N, E>;
        fn mul(self, rhs: Parser<'a, S, P, E>) -> Parser<'a, S, N, E> {
            rhs.sequence(self)
        }
    }

    impl<E> Add for ParseErr<E> {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            let mut errors = self.0;
            errors.extend(rhs.0);
            ParseErr(errors)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    use alloc::string::String;

    #[test]
    fn char() {
        let c: Parser<'_, Vec<char>, char, ()> = find_one('6');
        assert_eq!(
            c.run("69".chars().collect::<Vec<char>>())
                .map(|(s, c)| (s.into_iter().collect::<String>(), c)),
            Ok((String::from("9"), '6'))
        );
    }

    #[test]
    fn string_literal() {
        let string_literal: Parser<'_, Vec<char>, Vec<char>, ()> =
            find_one('"') >> find_predicate(|c| *c != '"') << find_one('"');

        assert_eq!(
            string_literal
                .run("\"Hello, World!\",".chars().collect::<Vec<char>>())
                .map(|(input, parsed)| (
                    input.into_iter().collect::<String>(),
                    parsed.into_iter().collect::<String>()
                )),
            Ok((",".into(), "Hello, World!".into()))
        )
    }
}
