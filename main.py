import argparse, os
from src.config import Configurator

if __name__ == '__main__':        
    parser = argparse.ArgumentParser(description='MRI Reconstruction')
    parser.add_argument(
        "--config", "-c", 
        type=str, 
        help="Path to config file"
    )
    args = parser.parse_args()

    # configurate and save configuration file
    cc = Configurator(args)
    os.makedirs(cc.cfg.exp_dir, exist_ok=True)
    with open(f'{cc.cfg.exp_dir}/config.yaml', 'w') as f:
        f.write(str(cc.cfg))
    
    exp, pl_module, data = cc.init_all()
    
    exp(pl_module, data)