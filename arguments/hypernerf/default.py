ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 150]
    },
    multires = [1,2,4,8],
    defor_depth = 3,
    net_width = 256,
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes =  0.001,
    no_grid = False,
    no_pe = True,
    no_do = True,
    pos_freq_bands=10,
    time_freq_bands=10

)
OptimizationParams = dict(
    dataloader=False,
    iterations = 60_000,
    coarse_iterations = 2000,
    densify_until_iter = 45_000,
    opacity_reset_interval = 6000,
    # position_lr_init = 0.00016,
    # position_lr_final = 0.0000016,
    # position_lr_delay_mult = 0.01,
    # position_lr_max_steps = 60_000,
    deformation_lr_init = 0.0016,
    deformation_lr_final = 0.00016,
    deformation_lr_delay_mult = 0.01,
    grid_lr_init = 0.016,
    grid_lr_final = 0.0016,
    # densify_until_iter = 50_000,
    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,
)
PipelineParams = dict(
    convert_SHs_python = True
)
# OptimizationParams = dict(
#     dataloader=False,
#     iterations = 30_000,
#     coarse_iterations = 3000,
#     densify_until_iter = 15_000,
#     opacity_reset_interval = 6000,
#     # position_lr_init = 0.00016,
#     # position_lr_final = 0.0000016,
#     # position_lr_delay_mult = 0.01,
#     # position_lr_max_steps = 60_000,
#     deformation_lr_init = 0.0016,
#     deformation_lr_final = 0.00016,
#     deformation_lr_delay_mult = 0.01,
#     grid_lr_init = 0.016,
#     grid_lr_final = 0.0016,
#     # densify_until_iter = 50_000,
#     opacity_threshold_coarse = 0.005,
#     opacity_threshold_fine_init = 0.005,
#     opacity_threshold_fine_after = 0.005,
#     # pruning_from_iter = 5000,
#     # pruning_interval = 8000
# )

# DyNeRF
# OptimizationParams = dict(

#     coarse_iterations = 3000,
#     deformation_lr_init = 0.00016,
#     deformation_lr_final = 0.0000016,
#     deformation_lr_delay_mult = 0.01,
#     grid_lr_init = 0.0016,
#     grid_lr_final = 0.000016,
#     iterations = 20000,
#     pruning_interval = 8000,
#     percent_dense = 0.01,
#     # opacity_reset_interval=30000

# )