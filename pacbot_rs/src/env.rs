use crate::grid::{coords_to_node, NODE_COORDS, VALID_ACTIONS};
use ndarray::{s, Array, Array3};
use num_enum::{IntoPrimitive, TryFromPrimitive};
use numpy::{IntoPyArray, PyArray3};
use pacbot_rs_2::game_modes::GameMode;
use pacbot_rs_2::game_state::GameState;
use pacbot_rs_2::location::{Direction, Direction::*, LocationState};
use pacbot_rs_2::variables::{self, GHOST_FRIGHT_STEPS, INIT_LEVEL};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::{seq::SliceRandom, Rng};
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct PacmanGymConfiguration {
    /// If true, other "random" options apply; in addition:
    /// - Pacman starting position is randomized
    /// - Sometimes some of the pellets are wiped from the board
    pub random_start: bool,
    /// If this && `random_start`, randomize the ghosts' starting positions
    pub randomize_ghosts: bool,

    /// If true, randomize pacman speed per game, else [`NORMAL_TICKS_PER_STEP`]
    pub random_ticks: bool,
    /// If `random_ticks`, the minimum number of game ticks per pacman action (randomized per game)
    ///
    /// - Default 4 (3 gu/s), meaning the game steps 4 times for each action Pacman takes
    /// - See [`NORMAL_TICKS_PER_STEP`]
    pub random_ticks_per_step_min: u32,
    /// If `random_ticks`, the maximum number of game ticks per pacman action (randomized per game)
    ///
    /// - Default 14 (~0.86 gu/s), meaning the game steps 14 times for each action Pacman takes
    /// - See [`NORMAL_TICKS_PER_STEP`]
    pub random_ticks_per_step_max: u32,

    /// If true, regular pellets should never appear in the observation, even if present in the game
    ///
    /// Note: super pellets may still appear
    pub obs_ignore_regular_pellets: bool,
    /// If true, super pellets should never appear in the observation, even if present in the game
    ///
    /// Note: regular pellets may still appear
    pub obs_ignore_super_pellets: bool,
}

impl Default for PacmanGymConfiguration {
    fn default() -> Self {
        Self {
            random_start: false,
            random_ticks: false,
            randomize_ghosts: false,

            random_ticks_per_step_min: 4,
            random_ticks_per_step_max: 14,

            obs_ignore_regular_pellets: false,
            obs_ignore_super_pellets: false,
        }
    }
}

pub const OBS_SHAPE: (usize, usize, usize) = (17, 28, 31);

pub const TICKS_PER_UPDATE: u32 = 12;
/// How many ticks the game should move every step normally. Ghosts move every 12 ticks.
///
/// - At 8 ticks/step (1.5 gu/s), the game steps 8 times for each action Pacman takes
/// - Lower ticks/step simulates a faster robot/easier game
/// - Higher ticks/step simulates a slower robot/harder game
/// - The game steps at 12 - 2 * ((level as i32) - 1) ticks/step
/// - At level 1 with 4 pacman ticks/step and 12 ghost ticks/step, pacman takes 3 actions for every 1 ghost actions
/// - Convert to average grid units per second using: 12 / ticks/step
const NORMAL_TICKS_PER_STEP: u32 = 8;

/// Penalty for turning.
const TURN_PENALTY: i32 = -2;

const LAST_REWARD: u16 = 3_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq, TryFromPrimitive, IntoPrimitive)]
#[repr(u8)]
pub enum Action {
    Stay = 0,
    Down = 1,
    Up = 2,
    Left = 3,
    Right = 4,
}

impl Action {
    /// Converts the given action index into an `Action`.
    ///
    /// Panics if index is outside the range `0..5`.
    pub fn from_index(index: usize) -> Self {
        Self::try_from_primitive(index.try_into().unwrap()).unwrap()
    }
}

impl From<Direction> for Action {
    fn from(value: Direction) -> Self {
        match value {
            Up => Action::Up,
            Left => Action::Left,
            Down => Action::Down,
            Right => Action::Right,
            Stay => Action::Stay,
        }
    }
}

impl From<Action> for Direction {
    fn from(value: Action) -> Self {
        match value {
            Action::Up => Up,
            Action::Left => Left,
            Action::Down => Down,
            Action::Right => Right,
            Action::Stay => Stay,
        }
    }
}

impl<'source> FromPyObject<'source> for Action {
    fn extract_bound(ob: &Bound<'source, PyAny>) -> PyResult<Self> {
        let ob = ob.as_borrowed();
        let index: u8 = ob.extract()?;
        Action::try_from_primitive(index).map_err(|_| PyValueError::new_err("Invalid action"))
    }
}

impl IntoPy<PyObject> for Action {
    fn into_py(self, py: Python<'_>) -> PyObject {
        u8::from(self).into_py(py)
    }
}

#[derive(Clone)]
#[pyclass]
pub struct PacmanGym {
    pub game_state: GameState,
    last_score: u16,
    last_action: Action,

    ticks_per_step: u32,

    pub config: PacmanGymConfiguration,
}

fn modify_bit_u32(num: &mut u32, bit_idx: usize, bit_val: bool) {
    // If the bit is true, we should set the bit, otherwise we clear it
    if bit_val {
        *num |= 1 << bit_idx;
    } else {
        *num &= !(1 << bit_idx);
    }
}

/// Converts game location into our coords.
fn loc_to_pos(loc: LocationState) -> Option<(usize, usize)> {
    if loc.row != 32 && loc.col != 32 {
        Some((loc.col as usize, (31 - loc.row - 1) as usize))
    } else {
        None
    }
}

#[pymethods]
impl PacmanGym {
    #[new]
    pub fn new(configuration: Py<PyAny>) -> Self {
        let game_state = GameState { paused: false, ..GameState::new() };
        let configuration = Python::with_gil(|py| {
            serde_pyobject::from_pyobject(configuration.bind(py).clone()).unwrap()
        });
        Self::new_with_state(configuration, game_state)
    }

    pub fn set_py_configuration(&mut self, configuration: Py<PyAny>) {
        let configuration = Python::with_gil(|py| {
            serde_pyobject::from_pyobject(configuration.bind(py).clone()).unwrap()
        });
        self.config = configuration;
    }

    pub fn reset(&mut self) {
        self.last_score = 0;
        self.game_state = GameState::new();

        let rng = &mut rand::thread_rng();

        if self.config.random_ticks {
            self.ticks_per_step = rng.gen_range(
                self.config.random_ticks_per_step_min..=self.config.random_ticks_per_step_max,
            );
        }

        if self.config.random_start {
            let mut random_pos = || *NODE_COORDS.choose(rng).unwrap();

            let pac_random_pos = random_pos();
            self.game_state.pacman_loc = LocationState::new(pac_random_pos.0, pac_random_pos.1, Up);

            if self.config.randomize_ghosts {
                for ghost in &mut self.game_state.ghosts {
                    ghost.trapped_steps = 0;
                    let ghost_random_pos = random_pos();
                    // find a valid next space
                    let index = coords_to_node(ghost_random_pos).expect("invalid random pos!");
                    let valid_moves = VALID_ACTIONS[index]
                        .iter()
                        .enumerate()
                        .filter(|(_, b)| **b)
                        .nth(1)
                        .unwrap();
                    ghost.loc = LocationState {
                        row: ghost_random_pos.0,
                        col: ghost_random_pos.1,
                        dir: match valid_moves.0 {
                            1 => Down,
                            2 => Up,
                            3 => Right,
                            4 => Left,
                            _ => unreachable!(),
                        },
                    };
                    ghost.next_loc = ghost.loc;
                }
            }

            // Randomly remove pellets from half the board (left, right, top, bottom) or don't.
            let wipe_type = rng.gen_range(0..=4);
            if wipe_type != 0 {
                for row in 0..31 {
                    for col in 0..28 {
                        if self.game_state.pellet_at((row, col))
                            && match wipe_type {
                                1 => col < 28 / 2,
                                2 => col >= 28 / 2,
                                3 => row < 31 / 2,
                                4 => row >= 31 / 2,
                                _ => unreachable!(),
                            }
                        {
                            modify_bit_u32(
                                &mut self.game_state.pellets[row as usize],
                                col as usize,
                                false,
                            );
                            self.game_state.decrement_num_pellets();
                        }
                    }
                }
            }
        }

        self.last_action = Action::Stay;
        self.game_state.paused = false;
    }

    /// Performs an action and steps the environment.
    /// Returns (reward, done).
    pub fn step(&mut self, action: Action) -> (i32, bool) {
        // Update Pacman pos
        self.move_one_cell(action);

        // step through environment multiple times
        let turn_penalty = if self.last_action == action || self.last_action == Action::Stay {
            0
        } else {
            TURN_PENALTY
        };
        for _ in 0..self.ticks_per_step {
            self.game_state.step();
            if self.is_done() {
                break;
            }
        }
        self.last_action = action;

        let game_state = &self.game_state;

        let done = self.is_done();

        // The reward is raw difference in game score, minus a penalty for dying or
        // plus a bonus for clearing the board.
        let mut reward = (game_state.curr_score as i32 - self.last_score as i32) + turn_penalty;
        if done {
            if game_state.curr_lives < 3 {
                // Pacman died.
                reward += -200;
            } else {
                // Pacman cleared the board! Good Pacman.
                reward += LAST_REWARD as i32;
            }
        }
        // Punishment for being too close to a ghost
        let pacbot_pos = self.game_state.pacman_loc;
        for ghost in &game_state.ghosts {
            if !ghost.is_frightened() {
                let ghost_pos = ghost.loc;
                reward += match (pacbot_pos.row - ghost_pos.row).abs()
                    + (pacbot_pos.col - ghost_pos.col).abs()
                {
                    1 => -50,
                    2 => -20,
                    _ => 0,
                };
            }
        }
        self.last_score = game_state.curr_score;

        (reward, done)
    }

    pub fn score(&self) -> u32 {
        self.game_state.curr_score as u32
    }

    pub fn lives(&self) -> u8 {
        self.game_state.get_lives()
    }

    pub fn is_done(&self) -> bool {
        self.game_state.get_lives() < 3 || self.game_state.get_level() != INIT_LEVEL
    }

    pub fn first_ai_done(&self) -> bool {
        // are super pellets gone and ghosts not frightened? then switch models
        [(3, 1), (3, 26), (23, 1), (23, 26)].into_iter().all(|x| !self.game_state.pellet_at(x))
            && self.game_state.ghosts.iter().all(|g| !g.is_frightened())
    }

    pub fn all_ghosts_freed(&self) -> bool {
        // are all ghosts freed from pen?
        self.game_state.ghosts.iter().all(|g| !g.is_trapped())
    }

    pub fn all_ghosts_not_frightened(&self) -> bool {
        // are all ghosts not frightened?
        self.game_state.ghosts.iter().all(|g| !g.is_frightened())
    }

    pub fn remaining_pellets(&self) -> u16 {
        self.game_state.get_num_pellets()
    }

    /// Returns the action mask that is `True` for currently-valid actions and
    /// `False` for currently-invalid actions.
    pub fn action_mask(&self) -> [bool; 5] {
        let p = self.game_state.pacman_loc;
        [
            true,
            !self.game_state.wall_at((p.row + 1, p.col)),
            !self.game_state.wall_at((p.row - 1, p.col)),
            !self.game_state.wall_at((p.row, p.col - 1)),
            !self.game_state.wall_at((p.row, p.col + 1)),
        ]
    }

    /// Returns an observation array/tensor constructed from the game state.
    pub fn obs_numpy(&self, py: Python<'_>) -> Py<PyArray3<f32>> {
        self.obs().into_pyarray_bound(py).into()
    }

    /// Prints a representation of the game state to standard output.
    pub fn print_game_state(&self) {
        // Print the score.
        print!("Score: {}", self.score());
        if self.is_done() {
            print!("  [DONE]");
        }
        println!();

        let game_state = &self.game_state;

        // Print the game grid.
        let ghost_char = |x, y| {
            for (i, ch) in ['R', 'P', 'B', 'O'].iter().enumerate() {
                if (x, y) == (game_state.ghosts[i].loc.row, game_state.ghosts[i].loc.col) {
                    let color =
                        if game_state.ghosts[i].is_frightened() { "96" } else { "38;5;206" };
                    return Some((*ch, color));
                }
            }
            None
        };
        for row in 0..31 {
            for col in 0..28 {
                let (ch, style) =
                    if (row, col) == (game_state.pacman_loc.row, game_state.pacman_loc.col) {
                        ('@', "93")
                    } else if let Some(ch) = ghost_char(row, col) {
                        ch
                    } else if game_state.wall_at((row, col)) {
                        ('#', "90")
                    } else if (row, col) == (game_state.fruit_loc.row, game_state.fruit_loc.col)
                        && game_state.fruit_exists()
                    {
                        ('c', "31")
                    } else if game_state.pellet_at((row, col)) {
                        if ((row == 3) || (row == 23)) && ((col == 1) || (col == 26)) {
                            // super pellet
                            ('o', "")
                        } else {
                            ('.', "")
                        }
                    } else {
                        (' ', "")
                    };
                print!("\x1b[{style}m{ch}\x1b[0m");
            }
            println!();
        }
    }
}

impl PacmanGym {
    pub fn new_with_config(configuration: PacmanGymConfiguration) -> Self {
        let game_state = GameState { paused: false, ..GameState::new() };
        Self::new_with_state(configuration, game_state)
    }

    pub fn new_with_state(configuration: PacmanGymConfiguration, game_state: GameState) -> Self {
        Self {
            last_score: 0,
            last_action: Action::Stay,
            game_state,
            ticks_per_step: NORMAL_TICKS_PER_STEP,

            config: configuration,
        }
    }

    pub fn set_state(&mut self, new_state: GameState, ticks_per_step: u32) {
        self.game_state = new_state;
        self.ticks_per_step = ticks_per_step;

        self.last_action = Action::Stay;
        self.last_score = self.game_state.curr_score;
    }

    fn move_one_cell(&mut self, action: Action) {
        let old_pos = self.game_state.pacman_loc;
        let new_pos = match action {
            Action::Stay => (old_pos.row, old_pos.col),
            Action::Right => (old_pos.row, old_pos.col + 1),
            Action::Left => (old_pos.row, old_pos.col.saturating_sub(1)),
            Action::Up => (old_pos.row.saturating_sub(1), old_pos.col),
            Action::Down => (old_pos.row + 1, old_pos.col),
        };
        if !self.game_state.wall_at(new_pos) {
            self.game_state.set_pacman_location(new_pos);
        }
    }

    /// Returns an observation array/tensor constructed from the game state.
    pub fn obs(&self) -> Array3<f32> {
        let game_state = &self.game_state;
        let mut obs_array = Array::zeros(OBS_SHAPE);
        let (mut wall, mut reward, mut pacman, mut ghost, mut last_ghost, mut state) = obs_array
            .multi_slice_mut((
                s![0, .., ..],
                s![1, .., ..],
                s![2..4, .., ..],
                s![4..8, .., ..],
                s![8..12, .., ..],
                s![12..15, .., ..],
            ));

        for row in 0..31 {
            for col in 0..28 {
                let obs_row = 31 - row - 1;
                wall[(col, obs_row)] = game_state.wall_at((row as i8, col as i8)) as u8 as f32;
                reward[(col, obs_row)] = if game_state.pellet_at((row as i8, col as i8)) {
                    (if ((row == 3) || (row == 23)) && ((col == 1) || (col == 26)) {
                        if self.config.obs_ignore_super_pellets {
                            0
                        } else {
                            variables::SUPER_PELLET_POINTS
                        }
                    } else {
                        if self.config.obs_ignore_regular_pellets {
                            0
                        } else {
                            variables::PELLET_POINTS
                        }
                    }) + (if game_state.num_pellets == 1 { LAST_REWARD } else { 0 })
                } else if game_state.fruit_exists()
                    && col == game_state.fruit_loc.col as usize
                    && row == game_state.fruit_loc.row as usize
                {
                    variables::FRUIT_POINTS
                } else {
                    0
                } as f32
                    / variables::COMBO_MULTIPLIER as f32;
            }
        }

        // Compute new pacman and ghost positions
        let new_pos = loc_to_pos(game_state.pacman_loc);
        let new_ghost_pos: Vec<_> = game_state.ghosts.iter().map(|g| loc_to_pos(g.loc)).collect();

        // last pos is not useful information during competition because our position changes are
        // continuous, not linked to discrete steps, which makes it difficult to track
        // in the future, the AI should be given the next position in the direction we are facing
        if let Some(last_pos) = /* self.last_pos */ new_pos {
            pacman[(0, last_pos.0, last_pos.1)] = 1.0;
        }
        if let Some(new_pos) = new_pos {
            pacman[(1, new_pos.0, new_pos.1)] = 1.0;
        }

        for (i, g) in game_state.ghosts.iter().enumerate() {
            if let Some((col, row)) = new_ghost_pos[i] {
                ghost[(i, col, row)] = 1.0;
                if g.is_frightened() {
                    state[(2, col, row)] = g.fright_steps as f32 / GHOST_FRIGHT_STEPS as f32;
                    reward[(col, row)] += 2_i32.pow(game_state.ghost_combo as u32) as f32;
                } else {
                    let state_index = if game_state.mode == GameMode::CHASE { 1 } else { 0 };
                    state[(state_index, col, row)] =
                        game_state.get_mode_steps() as f32 / GameMode::CHASE.duration() as f32;
                }
            }
        }

        // last ghost pos is not useful information because between most updates in real competition,
        // the ghosts don't move. In the future, the ai should be given the ghost's next position
        // based on the direction it is facing
        for (i, pos) in /* self.last_ghost_pos */ new_ghost_pos.iter().enumerate() {
            if let Some(pos) = pos {
                last_ghost[(i, pos.0, pos.1)] = 1.0;
            }
        }

        obs_array
            .slice_mut(s![15, .., ..])
            .fill(self.ticks_per_step as f32 / game_state.get_update_period() as f32);

        // Super pellet map
        for row in 0..31 {
            for col in 0..28 {
                let obs_row = 31 - row - 1;
                if !self.config.obs_ignore_super_pellets
                    && game_state.pellet_at((row as i8, col as i8))
                    && ((row == 3) || (row == 23))
                    && ((col == 1) || (col == 26))
                {
                    obs_array[(16, col, obs_row)] = 1.;
                }
            }
        }

        obs_array
    }
}
