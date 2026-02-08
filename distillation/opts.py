def model_opts(parser):
    # Feature embedding
    parser.add_argument("--pkt_len_vocab_size", type=int, default=1501)
    parser.add_argument("--pkt_len_embed_bits", type=int, default=10)
    parser.add_argument("--ipd_vocab_size", type=int, default=16001)
    parser.add_argument("--ipd_embed_bits", type=int, default=8)
    parser.add_argument("--embed_dim_bits", type=int, default=6)
    
    # RNN cell
    parser.add_argument("--seq_window_size", type=int, default=8)
    parser.add_argument("--pkts_per_rnn_step", type=int, default=1)
    parser.add_argument("--rnn_hidden_state_bits", type=int, default=9)
    # Output layer
    parser.add_argument("--num_classes", type=int, default=None)
    
    # Transformers
    parser.add_argument("--tfm_model_dim", type=int, default=8)
    parser.add_argument("--tfm_num_heads", type=int, default=4)
    parser.add_argument("--tfm_num_layers", type=int, default=4)

    # Loss
    parser.add_argument("--ce_loss_weight", type=float, default=0.8)
    parser.add_argument("--loss_scope", default="all", choices=["single", "all"],
                        help="If not set, use dataset default")
    parser.add_argument("--focal_gamma", type=float, default=0.0)



def training_opts(parser):
    # Steps
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--ckpt_save_interval_epochs", type=int, default=5)
    
    # Gpu
    parser.add_argument("--cuda_device_id", type=int, default=0)
    
    # Optimizer and scheduler
    parser.add_argument("--optim_name", default="adamw", choices=["adam", "adamw", "adafactor"])
    parser.add_argument("--lr", type=float, default=1e-2,)
    
    # Others
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--random_seed", type=int, default=7, 
                        help="Random seed.")


def aggregator_opts(parser):
    parser.add_argument("--quant_levels", type=int, default=16)
    parser.add_argument("--state_reset_cycle", type=int, default=128)


def simulation_opts(parser):
    parser.add_argument("--sim_duration_s", type=float, default=1)
    parser.add_argument("--flow_time_scale", type=int, default=8,
                        help="speed up the flow duration")
    parser.add_argument("--flow_rate_fps", type=int, default=160000,
                        help="network load")
    
    parser.add_argument('--flow_state_reset_ms', type=float, default=256,
                        help="refresh stated memory for a flow on switch")
    parser.add_argument('--imis_fallback_ratio', type=float, default=0.0,
                        help="the capacity of imis when two flows collided")
    
    parser.add_argument("--pkt_tree_count", type=int, default=2)
    parser.add_argument("--pkt_tree_depth", type=int, default=9)
    


def distill_opts(parser, require_teacher_ckpt=True):
    parser.add_argument("--teacher_ckpt_path", type=str, required=require_teacher_ckpt)
    parser.add_argument("--student_ckpt_path", type=str, default="")
    parser.add_argument("--kd_alpha", type=float, default=0.1)
    parser.add_argument("--kd_temperature", type=float, default=4.0)
