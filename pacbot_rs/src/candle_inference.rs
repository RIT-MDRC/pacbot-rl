use crate::candle_qnet::QNetV2;
use crate::env::{Action, PacmanGym, PacmanGymConfiguration, OBS_SHAPE};
use crate::grid::coords_to_node;
use candle_core::{Device, Module, Tensor, D};
use candle_nn as nn;
use pacbot_rs_2::game_state::GameState;
use pacbot_rs_2::location::Direction;

pub struct CandleInference {
    pub net: QNetV2,
    pub gym: PacmanGym,
    pub configuration: PacmanGymConfiguration,
}

impl CandleInference {
    /// Create a new `CandleInference` that can be used to get actions for a specific model
    pub fn new(weights_path: &str, configuration: PacmanGymConfiguration) -> Self {
        let mut vm = nn::VarMap::new();
        let vb =
            nn::VarBuilder::from_varmap(&vm, candle_core::DType::F32, &candle_core::Device::Cpu);
        let net = QNetV2::new(
            candle_core::Shape::from_dims(&[OBS_SHAPE.0, OBS_SHAPE.1, OBS_SHAPE.2]),
            5,
            vb,
        )
        .unwrap();
        vm.load(weights_path).unwrap();

        Self { net, gym: PacmanGym::new(&configuration), configuration }
    }

    /// Given a game state, find the reward the model predicts for each action. Indices available
    /// in [`Action`].
    ///
    /// If no action mask is provided, will use the one from training. A more advanced variant can
    /// be obtained from [`CandleInference::complex_action_mask`] which takes into account ghost
    /// locations.
    pub fn get_actions(
        &mut self,
        game_state: GameState,
        action_mask: Option<[bool; 5]>,
        ticks_per_step: u32,
    ) -> (Direction, [f32; 5]) {
        self.gym.set_state(game_state, ticks_per_step);

        let obs_array = self.gym.obs(&self.configuration);
        // 1 if masked, 0 if not
        let action_mask_arr = Tensor::from_slice(
            &action_mask.unwrap_or(self.gym.action_mask()).map(|b| b as u8 as f32),
            5,
            &Device::Cpu,
        )
        .unwrap();

        // Run observation through model and generate action.
        let obs_flat = obs_array.as_slice().unwrap();
        let obs_tensor = Tensor::from_slice(obs_flat, OBS_SHAPE, &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap()
            .to_dtype(candle_core::DType::F32)
            .unwrap();

        let q_vals = self.net.forward(&obs_tensor).unwrap().squeeze(0).unwrap();

        let q_vals = ((q_vals * &action_mask_arr).unwrap()
            + ((1. - &action_mask_arr).unwrap() * -999.).unwrap())
        .unwrap();
        let argmax_idx = q_vals.argmax(D::Minus1).unwrap().to_scalar::<u32>().unwrap() as usize;
        let mut argmax = [0.; 5];
        argmax[argmax_idx] = 1.;

        let actions = [Action::Stay, Action::Down, Action::Up, Action::Left, Action::Right];
        let chosen_action =
            actions[q_vals.argmax(D::Minus1).unwrap().to_scalar::<u32>().unwrap() as usize];
        let f32_q_vals = q_vals.to_vec1().unwrap();
        let direction_q_vals = [0, 1, 2, 3, 4].map(|direction_idx| {
            let direction = Direction::try_from(direction_idx).unwrap();
            let action: Action = direction.into();
            f32_q_vals[action as usize]
        });
        (chosen_action.into(), direction_q_vals)
    }

    /// Provides an action mask that takes into account ghost proximity.
    pub fn complex_action_mask(game_state: &GameState, scary_ghost_distance: i8) -> [bool; 5] {
        let mut action_mask = [false, false, false, false, false];
        let ghost_within = |row: i8, col: i8, distance: i8| {
            game_state.ghosts.iter().any(|g| {
                (g.loc.row - row).abs() + (g.loc.col - col).abs() <= distance && !g.is_frightened()
            })
        };
        let super_pellet_within = |row: i8, col: i8, distance: i8| {
            [(3, 1), (3, 26), (23, 1), (23, 26)].iter().any(|(p_row, p_col)| {
                (p_row - row).abs() + (p_col - col).abs() <= distance
                    && game_state.pellet_at((*p_row, *p_col))
            })
        };
        if coords_to_node((game_state.pacman_loc.row, game_state.pacman_loc.col)).is_some() {
            for ghost_deny_distance in (0..=scary_ghost_distance).rev() {
                let row = game_state.pacman_loc.row;
                let col = game_state.pacman_loc.col;
                action_mask =
                    [(row, col), (row + 1, col), (row - 1, col), (row, col - 1), (row, col + 1)]
                        .map(|(target_row, target_col)| {
                            !game_state.wall_at((target_row, target_col))
                                && (!ghost_within(target_row, target_col, ghost_deny_distance)
                                    || super_pellet_within(target_row, target_col, 0))
                        });
                if action_mask == [false; 5] {
                    action_mask[0] = true;
                }

                if action_mask != [true, false, false, false, false] {
                    break;
                }
            }
        }
        action_mask
    }
}
