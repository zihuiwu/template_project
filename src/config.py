import importlib
from fvcore.common.config import CfgNode as CN
from .experiment import Experiment
from .pl_module import PytorchLightningModule

class Configurator:
    def __init__(self, args):
        """
        Create configs and make fixes
        """
        self.cfg = CN(CN.load_yaml_with_base(args.config))

        # default fixes
        self.cfg = self._default_fix(self.cfg)

        # check cfg configurations
        self._check_cfg()
        
        self.cfg.freeze()
    
    def _check_cfg(self):
        pass

    def _default_fix(self, cfg):
        cfg.exp_dir =  f"results/{cfg.exp_name}"
        cfg.logger.name = cfg.exp_name
        cfg.logger.save_dir = cfg.exp_dir
        cfg.experiment.trainer.default_root_dir = cfg.exp_dir
        return cfg

    @staticmethod
    def str_to_class(module_name, class_name):
        """Return a class instance from a string reference"""
        try:
            module_ = importlib.import_module(module_name)
            try:
                class_ = getattr(module_, class_name)
            except AttributeError:
                raise AttributeError('Class does not exist')
        except ImportError:
            raise ImportError('Module does not exist')
        return class_

    @staticmethod
    def init_params_without_name(module_name, cfg):
        class_ = Configurator.str_to_class(module_name, cfg._class_)
        init_dict = dict(cfg)
        del init_dict["_class_"]
        return class_(**init_dict)

    def _init_data(self):
        return self.init_params_without_name("src.data", self.cfg.data)

    def _init_loss_fn(self):
        return self.init_params_without_name("src.metrics", self.cfg.loss_fn)

    def _init_metric_fns(self):
        return [self.init_params_without_name("src.metrics", v) for k, v in self.cfg.metric_fns.items()]

    def _init_model(self):
        return self.init_params_without_name("src.models", self.cfg.model)
    
    def _init_pl_module(self):
        model = self._init_model()
        loss_fn = self._init_loss_fn()
        metric_fns = self._init_metric_fns()

        pl_module = PytorchLightningModule(
            self.cfg,
            model,
            loss_fn,
            metric_fns
        )
        return pl_module

    def init_all(self):
        # exp
        exp = Experiment(self.cfg)

        # pytorch lightning module
        pl_module = self._init_pl_module()

        # data module
        data = self._init_data()

        return exp, pl_module, data