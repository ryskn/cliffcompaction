//! Terminal rendering helpers: colors, rules, and the wordmark banner.
//!
//! Everything degrades: no color when piped or when NO_COLOR is set, no
//! block glyphs when the terminal can't encode them or is too narrow.

use std::io::{IsTerminal, Write};

pub type Rgb = (u8, u8, u8);

pub const BRAND: Rgb = (0, 210, 190);
pub const YELLOW: Rgb = (255, 218, 51);
pub const TEXT: Rgb = (203, 213, 225);
pub const DIM: Rgb = (107, 122, 143);
pub const FAINT: Rgb = (58, 70, 84);
pub const IDLE: Rgb = (47, 110, 104);
pub const RED: Rgb = (248, 113, 113);

pub const MAX_WIDTH: usize = 104;

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_default()
}

/// Python's stdout encoding check, by way of the locale.
fn unicode_ok() -> bool {
    let locale = ["LC_ALL", "LC_CTYPE", "LANG"].iter().map(|k| env(k)).find(|v| !v.is_empty()).unwrap_or_default();
    let l = locale.to_lowercase();
    // Python runs the C/POSIX locale in UTF-8 mode, so those count too.
    l.is_empty() || l == "c" || l == "posix" || l.contains("utf-8") || l.contains("utf8")
}

/// `shutil.get_terminal_size((80, 24))`: COLUMNS/LINES, then the tty.
pub fn terminal_size() -> (usize, usize) {
    let from_env = |k: &str| env(k).trim().parse::<usize>().ok().filter(|n| *n > 0);
    let (mut cols, mut rows) = (from_env("COLUMNS"), from_env("LINES"));
    if cols.is_none() || rows.is_none() {
        #[cfg(unix)]
        {
            // SAFETY: TIOCGWINSZ writes a winsize into the struct we own.
            let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
            if unsafe { libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ, &mut ws) } == 0 {
                if ws.ws_col > 0 {
                    cols = cols.or(Some(ws.ws_col as usize));
                }
                if ws.ws_row > 0 {
                    rows = rows.or(Some(ws.ws_row as usize));
                }
            }
        }
    }
    (cols.unwrap_or(80), rows.unwrap_or(24))
}

/// Rendering context for stdout.
pub struct Term {
    pub tty: bool,
    pub color: bool,
    pub unicode: bool,
    pub rows: usize,
    pub raw_columns: usize,
    pub columns: usize,
}

impl Default for Term {
    fn default() -> Self {
        Term::new()
    }
}

impl Term {
    pub fn new() -> Term {
        let tty = std::io::stdout().is_terminal();
        let color = tty && env("NO_COLOR").is_empty() && env("TERM") != "dumb";
        let mut t = Term { tty, color, unicode: unicode_ok(), rows: 24, raw_columns: 80, columns: 80 };
        t.refresh();
        t
    }

    /// A plain context (no color, unicode on): tests and pipes.
    pub fn plain(columns: usize, rows: usize) -> Term {
        Term { tty: false, color: false, unicode: true, rows, raw_columns: columns, columns: columns.min(MAX_WIDTH) }
    }

    /// Re-read the terminal size (it changes when the window does).
    pub fn refresh(&mut self) {
        let (cols, rows) = terminal_size();
        self.raw_columns = cols;
        self.rows = rows;
        self.columns = cols.min(MAX_WIDTH);
    }

    pub fn c(&self, rgb: Rgb, s: &str) -> String {
        self.paint(rgb, s, false)
    }

    pub fn bold(&self, rgb: Rgb, s: &str) -> String {
        self.paint(rgb, s, true)
    }

    fn paint(&self, (r, g, b): Rgb, s: &str, bold: bool) -> String {
        if !self.color {
            return s.to_string();
        }
        let pre = if matches!(env("COLORTERM").as_str(), "truecolor" | "24bit") {
            format!("\x1b[38;2;{r};{g};{b}m")
        } else {
            let q = |v: u8| (f64::from(v) / 51.0).round() as u32;
            format!("\x1b[38;5;{}m", 16 + 36 * q(r) + 6 * q(g) + q(b))
        };
        format!("{pre}{}{s}\x1b[0m", if bold { "\x1b[1m" } else { "" })
    }

    pub fn rule(&self, label: &str) -> String {
        self.rule_width(label, self.columns)
    }

    pub fn rule_width(&self, label: &str, width: usize) -> String {
        let w = width.saturating_sub(2);
        let line = if self.unicode { "─" } else { "-" };
        if label.is_empty() {
            return format!(" {}", self.c(FAINT, &line.repeat(w)));
        }
        let head = format!("{} {label} ", line.repeat(2));
        let rest = w.saturating_sub(head.chars().count());
        format!(" {}", self.c(FAINT, &format!("{head}{}", line.repeat(rest))))
    }

    pub fn out(&self, lines: &[String]) {
        let mut out = std::io::stdout().lock();
        for line in lines {
            let _ = writeln!(out, "{line}");
        }
    }
}

// Wordmark glyphs, 6 rows tall; "#" stands for the block character.
const GLYPHS: &[(char, [&str; 6])] = &[
    ('C', [" ######╗ ", "##╔════╝ ", "##║      ", "##║      ", "╚######╗ ", " ╚═════╝ "]),
    ('L', ["##╗      ", "##║      ", "##║      ", "##║      ", "#######╗ ", "╚══════╝ "]),
    ('I', ["##╗ ", "##║ ", "##║ ", "##║ ", "##║ ", "╚═╝ "]),
    ('F', ["#######╗ ", "##╔════╝ ", "#####╗   ", "##╔══╝   ", "##║      ", "╚═╝      "]),
    ('O', [" ######╗  ", "##╔═══##╗ ", "##║   ##║ ", "##║   ##║ ", "╚######╔╝ ", " ╚═════╝  "]),
    ('M', ["###╗   ###╗ ", "####╗ ####║ ", "##╔####╔##║ ", "##║╚##╔╝##║ ", "##║ ╚═╝ ##║ ", "╚═╝     ╚═╝ "]),
    ('P', ["######╗  ", "##╔══##╗ ", "######╔╝ ", "##╔═══╝  ", "##║      ", "╚═╝      "]),
    ('A', [" #####╗  ", "##╔══##╗ ", "#######║ ", "##╔══##║ ", "##║  ##║ ", "╚═╝  ╚═╝ "]),
    ('T', ["########╗ ", "╚══##╔══╝ ", "   ##║    ", "   ##║    ", "   ##║    ", "   ╚═╝    "]),
    ('N', ["###╗   ##╗ ", "####╗  ##║ ", "##╔##╗ ##║ ", "##║╚##╗##║ ", "##║ ╚####║ ", "╚═╝  ╚═══╝ "]),
];

fn word(text: &str) -> Vec<String> {
    (0..6)
        .map(|r| {
            let row: String = text
                .chars()
                .map(|ch| GLYPHS.iter().find(|(c, _)| *c == ch).map_or("", |(_, g)| g[r]))
                .collect();
            row.replace('#', "█").trim_end().to_string()
        })
        .collect()
}

/// The CLIFFCOMPACTION wordmark, laid out for the terminal's width: one
/// line when it fits, stacked when it doesn't, plain text otherwise.
pub fn banner(term: &Term, indent: &str) -> Vec<String> {
    if term.unicode {
        let mut stacked = word("CLIFF");
        stacked.push(String::new());
        stacked.extend(word("COMPACTION"));
        for rows in [word("CLIFFCOMPACTION"), stacked] {
            let widest = rows.iter().map(|r| r.chars().count()).max().unwrap_or(0);
            if widest + indent.chars().count() <= term.raw_columns {
                let mut out = vec![String::new()];
                out.extend(rows.iter().map(|r| if r.is_empty() { String::new() } else { format!("{indent}{}", term.bold(BRAND, r)) }));
                out.push(String::new());
                return out;
            }
        }
    }
    vec![String::new(), format!("{indent}{}", term.bold(BRAND, "CLIFFCOMPACTION")), String::new()]
}
