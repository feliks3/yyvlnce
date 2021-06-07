import json
import jsonlines
import torch
import torch.nn as nn
import time
import math
import sys
sys.path.append('r2r_src')

from transformers import (BertConfig, BertTokenizer)
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from habitat_extensions.utils import observations_to_image
from habitat.utils.visualizations.utils import append_text_to_image
from habitat_baselines.common.environments import get_env_class

from habitat_baselines.utils.common import batch_obs, generate_video
import r2r_src.param
from habitat_baselines.utils.common import (
    get_checkpoint_id,
    poll_checkpoint_folder,
)
from vlnce_baselines.common.env_utils import (
    construct_envs_auto_reset_false,
    is_slurm_batch_job,
    construct_envs
)

import utils

from r2r_src.param import args

import model_OSCAR, model_PREVALENT
from collections import defaultdict
import matplotlib.pyplot as plt
from model.r2r_resnet_encoders import TorchVisionResNet152
from habitat_baselines.common.tensorboard_utils import TensorboardWriter

from typing import Dict, List

import os
from tensorboardX import SummaryWriter


import math
from vlnce_baselines.common.utils import extract_instruction_tokens
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat import logger
import tqdm
from vlnce_baselines.common import base_il_trainer
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from inspect import currentframe



def calculate_angle_feature(heading, elevation):

    import math
    # twopi = math.pi * 2
    # heading = (heading + twopi) % twopi     # From 0 ~ 2pi
    # It will be the same
    return np.array([math.sin(heading), math.cos(heading),
                     math.sin(elevation), math.cos(elevation)] * (128 // 4),
                    dtype=np.float32)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

class r2r_agent():

    def __init__(
        self, dagger_trainer, device, config, envs
    ):
        super().__init__()
        self.log_with_episode = ""
        self.dagger_trainer = dagger_trainer
        self.device = device
        self.envs = envs
        self.init_model()
        self.init_curr_elevation()

        if config != None:
            self.config = config
            self.init_directory()

        """
        self.observations = self.envs.reset()
        actions = torch.tensor([[2]], device=self.device)
        outputs = self.envs.step([a[0].item() for a in actions])
        print()
        """
        """
        1.
        输入当前视角
        输出feature和angle的矩阵，1*5*2176
        """
    def init_curr_elevation(self):
        if self.envs != None:
            self.curr_elevation = [1 for _ in range(self.envs.num_envs)]
            for i in range(self.envs.num_envs):
                self.reset_curr_elevation(i)

    def init_directory(self):

        self.log_dir = 'snap/%s' % self.config.R2R_MODEL.name
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def init_model(self):

        observation_rgb_size = [480, 640, 3]
        output_size = 2048
        self.model = TorchVisionResNet152(observation_rgb_size, output_size, self.device)
        self.model = self.model.cuda(self.device)
        #self.tok = tok
        #self.episode_len = episode_len
        self.feature_size = 2048

        # Models
        if args.vlnbert == 'oscar':
            self.vln_bert = model_OSCAR.VLNBERT(feature_size=self.feature_size + args.angle_feat_size).cuda(self.device)
            self.critic = model_OSCAR.Critic().cuda(self.device)
        elif args.vlnbert == 'prevalent':
            self.vln_bert = model_PREVALENT.VLNBERT(feature_size=self.feature_size + args.angle_feat_size).cuda(self.device)
            self.critic = model_PREVALENT.Critic().cuda(self.device)
        self.models = (self.vln_bert, self.critic)

        # Optimizers
        self.vln_bert_optimizer = args.optimizer(self.vln_bert.parameters(), lr=args.lr)
        self.critic_optimizer = args.optimizer(self.critic.parameters(), lr=args.lr)
        self.optimizers = (self.vln_bert_optimizer, self.critic_optimizer)

        # Evaluations
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignoreid, size_average=False)
        #self.ndtw_criterion = utils.ndtw_initialize()

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)


    def set_curr_elevation(self, index, target_action):
        pass
        """
        logger.info(f"set_curr_elevation(self, {index}, {target_action}),{currentframe().f_back.f_lineno},{self.log_with_episode}")
        # elevation 0:look up, 1:look forward, 2:look down
        if target_action == 1 or target_action == 2 or target_action == 3:
            pass
        elif target_action == 4:
            if self.curr_elevation[index] > 0:
                self.curr_elevation[index] -= 1
        elif target_action == 5:
            if self.curr_elevation[index] < 2:
                self.curr_elevation[index] += 1
        """

    def reset_curr_elevation(self, index):
        pass
        """
        logger.info(f"reset_curr_elevation(self, {index}),{currentframe().f_back.f_lineno},{self.log_with_episode}")
        self.curr_elevation[index] = 1
        """


    def set_instruction(self, instruction):
        self.instruction = instruction
        self.instr_encoding = self.encode_instr(self.instruction)

    def get_instruction(self):
        return self.instruction

    def get_instr_encoding(self):
        return self.instr_encoding

    def train_listener(self, train_ml=None, train_rl=True, reset=True, **kwargs):

        for epoch in range(self.config.R2R_MODEL.epoch):
            self.envs = construct_envs(self.config, get_env_class(self.config.ENV_NAME))

            self.observations = self.envs.reset()

            self.logs = defaultdict(list)
            writer = SummaryWriter(log_dir=self.log_dir)

            record_file = open('./logs/' + self.config.R2R_MODEL.name + '.txt', 'a')
            record_file.write(str(args) + '\n\n')
            record_file.close()

            start_iter = 0

            start = time.time()
            print('\nListener training starts, start iteration: %s' % str(
                start_iter))

            if self.config.R2R_MODEL.load != "None":
                start_iter = self.load(os.path.join(self.config.R2R_MODEL.load))
                print("\nLOAD the model from {}, iteration ".format(args.load, start_iter))



            """
            with TensorboardWriter(
                self.config.TENSORBOARD_DIR,
                flush_secs=5,
                purge_step=0,
            ) as writer:

                #self.dagger_trainer._eval_checkpoint(self.config.R2R_MODEL.load, writer, checkpoint_index=0)
            """


            #best_val = {'val_unseen': {"spl": 0., "sr": 0., "state": "", 'update': False}}

            for idx in range(start_iter, start_iter + self.config.R2R_MODEL.n_iters, self.config.R2R_MODEL.log_every):
                interval = min(self.config.R2R_MODEL.log_every, self.config.R2R_MODEL.n_iters - idx)
                iter = idx + interval

                jdx_length = len(range(interval // 2))
                for jdx in range(interval // 2):
                    # Train with GT data
                    args.ml_weight = 0.2
                    #self.train(1, feedback=feedback_method)
                    self.train(self.envs, self.config, train_ml=None, train_rl=True, reset=True, **kwargs)

                    print_progress(jdx, jdx_length, prefix='Progress:', suffix='Complete', bar_length=50)

                # Log the training stats to tensorboard
                total = max(sum(self.logs['total']), 1)
                length = max(len(self.logs['critic_loss']), 1)
                critic_loss = sum(self.logs['critic_loss']) / total
                RL_loss = sum(self.logs['RL_loss']) / max(
                    len(self.logs['RL_loss']), 1)
                IL_loss = sum(self.logs['IL_loss']) / max(
                    len(self.logs['IL_loss']), 1)
                entropy = sum(self.logs['entropy']) / total
                writer.add_scalar("loss/critic", critic_loss, idx)
                writer.add_scalar("policy_entropy", entropy, idx)
                writer.add_scalar("loss/RL_loss", RL_loss, idx)
                writer.add_scalar("loss/IL_loss", IL_loss, idx)
                writer.add_scalar("total_actions", total, idx)
                writer.add_scalar("max_length", length, idx)
                # print("total_actions", total, ", max_length", length)

                """
                # Run validation
                loss_str = "iter {}".format(iter)
                for env_name, (env, evaluator) in val_envs.items():

                    # Get validation distance from goal under test evaluation conditions
                    self.test(use_dropout=False, feedback='argmax', iters=None)
                    result = self.get_results()
                    score_summary, _ = evaluator.score(result)
                    loss_str += ", %s " % env_name
                    for metric, val in score_summary.items():
                        if metric in ['spl']:
                            writer.add_scalar("spl/%s" % env_name, val, idx)
                            if env_name in best_val:
                                if val > best_val[env_name]['spl']:
                                    best_val[env_name]['spl'] = val
                                    best_val[env_name]['update'] = True
                                elif (val == best_val[env_name]['spl']) and (
                                    score_summary['success_rate'] >
                                    best_val[env_name]['sr']):
                                    best_val[env_name]['spl'] = val
                                    best_val[env_name]['update'] = True
                        loss_str += ', %s: %.4f' % (metric, val)

                record_file = open('./logs/' + args.name + '.txt', 'a')
                record_file.write(loss_str + '\n')
                record_file.close()

                for env_name in best_val:
                    if best_val[env_name]['update']:
                        best_val[env_name]['state'] = 'Iter %d %s' % (
                        iter, loss_str)
                        best_val[env_name]['update'] = False
                        self.save(idx,
                                     os.path.join("snap", args.name, "state_dict",
                                                  "best_%s" % (env_name)))
                    else:
                        self.save(idx,
                                     os.path.join("snap", args.name, "state_dict",
                                                  "latest_dict"))

                print(
                    ('%s (%d %d%%) %s' % (timeSince(start, float(iter) / n_iters),
                                          iter, float(iter) / n_iters * 100,
                                          loss_str)))

                if iter % 1000 == 0:
                    print("BEST RESULT TILL NOW")
                    for env_name in best_val:
                        print(env_name, best_val[env_name]['state'])

                        record_file = open('./logs/' + args.name + '.txt', 'a')
                        record_file.write(
                            'BEST RESULT TILL NOW: ' + env_name + ' | ' +
                            best_val[env_name]['state'] + '\n')
                        record_file.close()

                """
                self.save(idx, os.path.join("snap", self.config.R2R_MODEL.name, "state_dict", "LAST_iter%d_%d" % (epoch,idx)))


    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def test(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = traj['path']
        else:   # Do a full round
            while True:
                for traj in self.rollout(**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = traj['path']
                if looped:
                    break

    def encode_instruction(self):
        current_episodes = self.envs.current_episodes()

        look_left = 2
        look_right = 3
        look_left_arr = [[look_left] for _ in range(self.envs.num_envs)]
        look_right_arr = [[look_right] for _ in range(self.envs.num_envs)]

        actions = torch.tensor(look_left_arr, device=self.device)
        outputs = self.envs.step([a[0].item() for a in actions])

        actions = torch.tensor(look_right_arr, device=self.device)
        outputs = self.envs.step([a[0].item() for a in actions])

        self.observations, reward, dones, info = [list(x) for x in
                                             zip(*outputs)]

        #print("instruction")
        instr_encoding = []
        for sig_episode in current_episodes:
            instr_encoding.append(self.encode_instr(sig_episode.instruction.instruction_text))

        # Language input
        sentence, language_attention_mask, token_type_ids, seq_lengths, perm_idx = self._sort_batch(
            instr_encoding)
        ''' Language BERT '''
        language_inputs = {'mode': 'language',
                           'sentence': sentence,
                           'attention_mask': language_attention_mask,
                           'lang_mask': language_attention_mask,
                           'token_type_ids': token_type_ids}

        h_t, language_features = self.vln_bert(**language_inputs)
        return h_t, language_features, language_attention_mask, token_type_ids, current_episodes, info

    def predict_next_step(self, h_t, language_features, language_attention_mask, token_type_ids):
        # if t > 10:
        # break

        env = None
        curr_elevation = None
        # print(t)
        # print(self.envs.current_episodes())
        input_a_t, candidate_feat, candidate_leng, candidates_mask = self.get_input_feat(
            env, curr_elevation)

        # if (t >= 1) or (args.vlnbert == 'prevalent'):
        language_features = torch.cat(
            (h_t.unsqueeze(1), language_features[:, 1:, :]), dim=1)

        visual_temp_mask = candidates_mask
        # (utils.length2mask(candidate_leng) == 0).long()
        visual_attention_mask = torch.cat(
            (language_attention_mask, visual_temp_mask), dim=-1)

        self.vln_bert.vln_bert.config.directions = max(
            candidate_leng)
        ''' Visual BERT '''
        visual_inputs = {'mode': 'visual',
                         'sentence': language_features,
                         'attention_mask': visual_attention_mask,
                         'lang_mask': language_attention_mask,
                         'vis_mask': visual_temp_mask,
                         'token_type_ids': token_type_ids,
                         'action_feats': input_a_t,
                         # 'pano_feats':         f_t,
                         'cand_feats': candidate_feat}
        h_t, logit = self.vln_bert(**visual_inputs)
        # actions = logit.max(1)
        _, actions = logit.max(1)
        """
        actions, rnn_states = self.policy.act(
            batch,
            rnn_states,
            prev_actions,
            not_done_masks,
            deterministic=not self.config.INFERENCE.SAMPLE,
        )
        prev_actions.copy_(actions)
        """

        actions = torch.tensor(actions, device=self.device)

        return actions, h_t, logit, candidates_mask

    def eval(self):
        logger.info(f"eval,{currentframe().f_back.f_lineno}")
        r"""Main method of trainer evaluation. Calls _eval_checkpoint() that
        is specified in Trainer class that inherits from BaseRLTrainer
        or BaseILTrainer

        Returns:
            None
        """
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if "tensorboard" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.TENSORBOARD_DIR) > 0
            ), "Must specify a tensorboard directory for video display"
            os.makedirs(self.config.TENSORBOARD_DIR, exist_ok=True)
        if "disk" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.VIDEO_DIR) > 0
            ), "Must specify a directory for storing videos on disk"

        self.flush_secs = 5
        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            if os.path.isfile(self.config.EVAL_CKPT_PATH_DIR):
                # evaluate singe checkpoint
                proposed_index = get_checkpoint_id(
                    self.config.EVAL_CKPT_PATH_DIR
                )
                if proposed_index is not None:
                    ckpt_idx = proposed_index
                else:
                    ckpt_idx = 0
                self._eval_checkpoint(
                    self.config.EVAL_CKPT_PATH_DIR,
                    writer,
                    checkpoint_index=ckpt_idx,
                )
            else:
                # evaluate multiple checkpoints in order
                prev_ckpt_ind = -1
                while True:
                    current_ckpt = None
                    while current_ckpt is None:
                        current_ckpt = poll_checkpoint_folder(
                            self.config.EVAL_CKPT_PATH_DIR, prev_ckpt_ind
                        )
                        time.sleep(2)  # sleep for 2 secs before polling again
                    logger.info(f"=======current_ckpt: {current_ckpt}=======")
                    prev_ckpt_ind += 1
                    self._eval_checkpoint(
                        checkpoint_path=current_ckpt,
                        writer=writer,
                        checkpoint_index=prev_ckpt_ind,
                    )

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        logger.info(f"_eval_checkpoint,{currentframe().f_back.f_lineno}")
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object
            checkpoint_index: index of the current checkpoint

        Returns:
            None
        """
        logger.info(f"checkpoint_path: {checkpoint_path}")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(
                self.load_checkpoint(checkpoint_path, map_location="cpu")[
                    "config"
                ]
            )
        else:
            config = self.config.clone()

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        config.TASK_CONFIG.DATASET.LANGUAGES = config.EVAL.LANGUAGES
        config.TASK_CONFIG.TASK.NDTW.SPLIT = config.EVAL.SPLIT
        config.TASK_CONFIG.TASK.SDTW.SPLIT = config.EVAL.SPLIT
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        config.IL.ckpt_to_load = checkpoint_path
        config.use_pbar = not is_slurm_batch_job()

        if len(config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")

        config.freeze()

        if config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                config.RESULTS_DIR,
                f"stats_ckpt_{checkpoint_index}_{config.TASK_CONFIG.DATASET.SPLIT}.json",
            )
            if os.path.exists(fname):
                print("skipping -- evaluation exists.")
                return

        """
        self.envs = construct_envs_auto_reset_false(
            config, get_env_class(config.ENV_NAME)
        )
        """
        self.envs = construct_envs(
            config, get_env_class(config.ENV_NAME)
        )


        self.init_curr_elevation()

        obs_transforms = get_active_obs_transforms(config)
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )

        """
        self._initialize_policy(
            config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=envs.action_spaces[0],
        )
        self.policy.eval()
        """

        self.observations = self.envs.reset()
        self.observations = extract_instruction_tokens(
            self.observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        batch = batch_obs(self.observations, self.device)
        batch = apply_obs_transforms_batch(batch, obs_transforms)

        num_recurrent_layers = 2
        rnn_states = torch.zeros(
            self.envs.num_envs,
            #self.policy.net.num_recurrent_layers,
            num_recurrent_layers,
            config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.envs.num_envs, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.envs.num_envs, 1, dtype=torch.uint8, device=self.device
        )

        stats_episodes = {}

        rgb_frames = [[] for _ in range(self.envs.num_envs)]
        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)

        if config.EVAL.EPISODE_COUNT == -1:
            episodes_to_eval = sum(self.envs.number_of_episodes)
        else:
            episodes_to_eval = min(
                config.EVAL.EPISODE_COUNT, sum(self.envs.number_of_episodes)
            )

        pbar = tqdm.tqdm(total=episodes_to_eval) if config.use_pbar else None
        log_str = (
            f"[Ckpt: {checkpoint_index}]"
            " [Episodes evaluated: {evaluated}/{total}]"
            " [Time elapsed (s): {time}]"
        )
        start_time = time.time()

        #while self.envs.num_envs > 0 and len(stats_episodes) < episodes_to_eval:
        for i in range(3):
            current_episodes = self.envs.current_episodes()

            """
            with torch.no_grad():
                actions, rnn_states = self.policy.act(
                    batch,
                    rnn_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=not config.EVAL.SAMPLE,
                )
                prev_actions.copy_(actions)
            """
            #curr_r2r_agent = r2r_agent(self, self.device, self.config, envs)
            h_t, language_features, language_attention_mask, token_type_ids, current_episodes, info = self.encode_instruction()
            print("i", i)
            # --------
            for t in range(self.config.R2R_MODEL.max_step):
                print(t, end=",")

                actions, h_t, logit, candidates_mask = self.predict_next_step(h_t, language_features, language_attention_mask, token_type_ids)
                actions = torch.tensor(
                    torch.reshape(actions, (actions.shape[0], 1)),
                    device=self.device)
                outputs = self.envs.step([a[0].item() for a in actions])
                self.observations, _, dones, infos = [list(x) for x in zip(*outputs)]
                #if dones[0] == True:
                    #break

                not_done_masks = torch.tensor(
                    [[0] if done else [1] for done in dones],
                    dtype=torch.uint8,
                    device=self.device,
                )

                # reset envs and observations if necessary
                for i in range(self.envs.num_envs):
                    if len(config.VIDEO_OPTION) > 0:
                        frame = self.observations_to_image(self.observations[i], infos[i])
                        frame = append_text_to_image(
                            frame, current_episodes[i].instruction.instruction_text
                        )
                        rgb_frames[i].append(frame)

                    #print(infos[0]["distance_to_goal"], dones)
                    if not dones[i] and t != self.config.R2R_MODEL.max_step:
                        continue

                    stats_episodes[current_episodes[i].episode_id] = infos[i]
                    self.observations[i] = self.envs.reset_at(i)[0]
                    prev_actions[i] = torch.zeros(1, dtype=torch.long)

                    if config.use_pbar:
                        pbar.update()
                    else:
                        logger.info(
                            log_str.format(
                                evaluated=len(stats_episodes),
                                total=episodes_to_eval,
                                time=round(time.time() - start_time),
                            )
                        )

                    if len(config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=config.VIDEO_OPTION,
                            video_dir=config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics={
                                "spl": stats_episodes[
                                    current_episodes[i].episode_id
                                ]["spl"]
                            },
                            tb_writer=writer,
                        )

                        del stats_episodes[current_episodes[i].episode_id][
                            "top_down_map_vlnce"
                        ]
                        del stats_episodes[current_episodes[i].episode_id][
                            "collisions"
                        ]
                        rgb_frames[i] = []

                self.observations = extract_instruction_tokens(
                    self.observations,
                    self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                )
                batch = batch_obs(self.observations, self.device)
                batch = apply_obs_transforms_batch(batch, obs_transforms)

                envs_to_pause = []
                next_episodes = self.envs.current_episodes()

                for i in range(self.envs.num_envs):
                    if next_episodes[i].episode_id in stats_episodes:
                        envs_to_pause.append(i)

                (
                    self.envs,
                    rnn_states,
                    not_done_masks,
                    prev_actions,
                    batch,
                    rgb_frames,
                ) = self.dagger_trainer._pause_envs(
                    envs_to_pause,
                    self.envs,
                    rnn_states,
                    not_done_masks,
                    prev_actions,
                    batch,
                    rgb_frames,
                )
                if dones[i] or t == self.config.R2R_MODEL.max_step:
                    break

        self.envs.close()
        if config.use_pbar:
            pbar.close()

        aggregated_stats = {}
        num_episodes = len(stats_episodes)
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values())
                / num_episodes
            )

        split = config.TASK_CONFIG.DATASET.SPLIT
        if config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                config.RESULTS_DIR,
                f"stats_ckpt_{checkpoint_index}_{split}.json",
            )
            with open(fname, "w") as f:
                json.dump(aggregated_stats, f, indent=4)

        logger.info(f"Episodes evaluated: {num_episodes}")
        checkpoint_num = checkpoint_index + 1
        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.6f}")
            writer.add_scalar(f"eval_{split}_{k}", v, checkpoint_num)



    def inference(self):
        logger.info(f"inference,{currentframe().f_back.f_lineno}")

        self.observations = self.envs.reset()


        obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )

        """
        self._initialize_policy(
            self.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
        )
        self.policy.eval()
        """

        observations = self.envs.reset()
        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, obs_transforms)

        num_recurrent_layers = 2
        rnn_states = torch.zeros(
            self.envs.num_envs,
            num_recurrent_layers,
            self.config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )


        prev_actions = torch.zeros(
            self.envs.num_envs, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.envs.num_envs, 1, dtype=torch.uint8, device=self.device
        )

        episode_predictions = defaultdict(list)

        # episode ID --> instruction ID for rxr predictions format
        instruction_ids: Dict[str, int] = {}

        # populate episode_predictions with the starting state
        current_episodes = self.envs.current_episodes()
        for i in range(self.envs.num_envs):
            episode_predictions[current_episodes[i].episode_id].append(
                self.envs.call_at(i, "get_info", {"observations": {}})
            )
            if self.config.INFERENCE.FORMAT == "rxr":
                ep_id = current_episodes[i].episode_id
                k = current_episodes[i].instruction.instruction_id
                instruction_ids[ep_id] = int(k)

        with tqdm.tqdm(
            total=sum(self.envs.count_episodes()),
            desc=f"[inference:{self.config.INFERENCE.SPLIT}]",
        ) as pbar:
            while self.envs.num_envs > 0:
                with torch.no_grad():
                    current_episodes = self.envs.current_episodes()

                    actions = torch.tensor([[2]], device=self.device)
                    outputs = self.envs.step([a[0].item() for a in actions])

                    actions = torch.tensor([[3]], device=self.device)
                    outputs = self.envs.step([a[0].item() for a in actions])

                    observations, reward, dones, info = [list(x) for x in
                                                         zip(*outputs)]

                    #print("instruction")
                    instruction = current_episodes[
                        0].instruction.instruction_text
                    instr_encoding = self.encode_instr(instruction)

                    # Language input
                    sentence, language_attention_mask, token_type_ids, seq_lengths, perm_idx = self._sort_batch(
                        instr_encoding)
                    ''' Language BERT '''
                    language_inputs = {'mode': 'language',
                                       'sentence': sentence,
                                       'attention_mask': language_attention_mask,
                                       'lang_mask': language_attention_mask,
                                       'token_type_ids': token_type_ids}

                    h_t, language_features = self.vln_bert(**language_inputs)

                    #--------
                    for t in range(3000):

                        #if t > 10:
                            #break

                        env = None
                        curr_elevation = None
                        #print(t)
                        #print(self.envs.current_episodes())
                        input_a_t, candidate_feat, candidate_leng, candidates_mask = self.get_input_feat(
                            env, curr_elevation)

                        if (t >= 1) or (args.vlnbert == 'prevalent'):
                            language_features = torch.cat(
                                (h_t.unsqueeze(1), language_features[:, 1:, :]),
                                dim=1)

                        visual_temp_mask = candidates_mask
                        # (utils.length2mask(candidate_leng) == 0).long()
                        visual_attention_mask = torch.cat(
                            (language_attention_mask, visual_temp_mask), dim=-1)

                        self.vln_bert.vln_bert.config.directions = max(
                            candidate_leng)
                        ''' Visual BERT '''
                        visual_inputs = {'mode': 'visual',
                                         'sentence': language_features,
                                         'attention_mask': visual_attention_mask,
                                         'lang_mask': language_attention_mask,
                                         'vis_mask': visual_temp_mask,
                                         'token_type_ids': token_type_ids,
                                         'action_feats': input_a_t,
                                         # 'pano_feats':         f_t,
                                         'cand_feats': candidate_feat}
                        h_t, logit = self.vln_bert(**visual_inputs)
                        #actions = logit.max(1)
                        _, actions = logit.max(1)
                        """
                        actions, rnn_states = self.policy.act(
                            batch,
                            rnn_states,
                            prev_actions,
                            not_done_masks,
                            deterministic=not self.config.INFERENCE.SAMPLE,
                        )
                        prev_actions.copy_(actions)
                        """


                        actions = torch.tensor([[actions]], device=self.device)

                        #print("get_done", self.envs.call_at(i, "get_done", {"observations": self.observations}))
                        outputs = self.envs.step([a[0].item() for a in actions])
                        self.observations, _, dones, infos = [
                            list(x) for x in zip(*outputs)
                        ]
                        print("action", actions, dones)
                        #print("dones" , dones)

                        not_done_masks = torch.tensor(
                            [[0] if done else [1] for done in dones],
                            dtype=torch.uint8,
                            device=self.device,
                        )

                        # reset envs and observations if necessary
                        for i in range(self.envs.num_envs):
                            episode_predictions[current_episodes[i].episode_id].append(
                                infos[i]
                            )
                            if not dones[i]:
                                continue

                            self.observations[i] = self.envs.reset_at(i)[0]
                            prev_actions[i] = torch.zeros(1, dtype=torch.long)
                            pbar.update()

                        self.observations = extract_instruction_tokens(
                            self.observations,
                            self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                        )
                        batch = batch_obs(self.observations, self.device)
                        batch = apply_obs_transforms_batch(batch, obs_transforms)

                        envs_to_pause = []
                        next_episodes = self.envs.current_episodes()
                        for i in range(self.envs.num_envs):
                            if not dones[i]:
                                continue

                            if next_episodes[i].episode_id in episode_predictions:
                                envs_to_pause.append(i)
                            else:
                                episode_predictions[
                                    next_episodes[i].episode_id
                                ].append(
                                    self.envs.call_at(i, "get_info", {"observations": {}})
                                )
                                if self.config.INFERENCE.FORMAT == "rxr":
                                    ep_id = next_episodes[i].episode_id
                                    k = next_episodes[i].instruction.instruction_id
                                    instruction_ids[ep_id] = int(k)

                        (
                            self.envs,
                            rnn_states,
                            not_done_masks,
                            prev_actions,
                            batch,
                            _,
                        ) = self.dagger_trainer._pause_envs(
                            envs_to_pause,
                            self.envs,
                            rnn_states,
                            not_done_masks,
                            prev_actions,
                            batch,
                        )
                break
        self.envs.close()

        if self.config.INFERENCE.FORMAT == "r2r":
            with open(self.config.INFERENCE.PREDICTIONS_FILE, "w") as f:
                json.dump(episode_predictions, f, indent=2)

            logger.info(
                f"Predictions saved to: {self.config.INFERENCE.PREDICTIONS_FILE}"
            )
        else:  # use 'rxr' format for rxr-habitat leaderboard
            predictions_out = []

            for k,v in episode_predictions.items():

                # save only positions that changed
                path = [v[0]["position"]]
                for p in v[1:]:
                    if path[-1] != p["position"]:
                        path.append(p["position"])

                predictions_out.append(
                    {
                        "instruction_id": instruction_ids[k],
                        "path": path,
                    }
                )

            predictions_out.sort(key=lambda x: x["instruction_id"])
            with jsonlines.open(
                self.config.INFERENCE.PREDICTIONS_FILE, mode="w"
            ) as writer:
                writer.write_all(predictions_out)

            logger.info(
                f"Predictions saved to: {self.config.INFERENCE.PREDICTIONS_FILE}"
            )

    def get_results(self):
        output = [{'instr_id': k, 'trajectory': v} for k, v in self.results.items()]
        return output

    def train(self, envs, config, train_ml=None, train_rl=True, reset=True, **kwargs):
        self.envs = envs
        feedback = "sample"
        n_iters = 1
        self.feedback = feedback

        self.vln_bert.train()
        self.critic.train()

        self.losses = []


        for iter in range(1, n_iters + 1):

            self.vln_bert_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            self.loss = 0

            if feedback == 'teacher':
                self.feedback = 'teacher'
                self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
            elif feedback == 'sample':  # agents in IL and RL separately
                if args.ml_weight != 0:
                    self.feedback = 'teacher'
                    #self.rollout(train_ml=args.ml_weight, train_rl=False, **kwargs)
                    self.rollout(envs, config, train_rl=False)

                self.feedback = 'sample'
                #self.rollout(train_ml=None, train_rl=True, **kwargs)
                self.rollout(envs, config, train_rl=True)

            else:
                assert False

            logger.info(f"train: iter2 1{iter},{currentframe().f_back.f_lineno},{self.log_with_episode}")
            self.loss.backward()
            logger.info(f"train: iter2 2{iter},{currentframe().f_back.f_lineno},{self.log_with_episode}")

            torch.nn.utils.clip_grad_norm(self.vln_bert.parameters(), 40.)

            self.vln_bert_optimizer.step()
            self.critic_optimizer.step()



    def rollout(self, envs, config, train_ml=None, train_rl=True, reset=True):

        self.envs = envs
        self.config = config
        expert_uuid = self.config.IL.DAGGER.expert_policy_sensor_uuid
        ml_loss = 0.

        #observations = self.envs.reset()

        rollout_count = 1
        for i in range(rollout_count):
            # 几个episode

            #self.reset()
            """
            current_episodes = self.envs.current_episodes()
            look_left = 2
            look_right = 3
            look_left_arr = [[look_left] for _ in range(self.envs.num_envs)]
            look_right_arr = [[look_right] for _ in range(self.envs.num_envs)]
            actions = torch.tensor(look_left_arr, device=self.device)
            outputs = self.envs.step([a[0].item() for a in actions])

            actions = torch.tensor(look_right_arr, device=self.device)
            outputs = self.envs.step([a[0].item() for a in actions])

            observations, reward, dones, info = [list(x) for x in zip(*outputs)]


            #print("instruction")
            instr_encoding = []
            for sig_episode in current_episodes:
                instr_encoding.append(self.encode_instr(sig_episode.instruction.instruction_text))

            # Language input
            sentence, language_attention_mask, token_type_ids, seq_lengths, perm_idx = self._sort_batch(
                instr_encoding)
            ''' Language BERT '''
            language_inputs = {'mode': 'language',
                               'sentence': sentence,
                               'attention_mask': language_attention_mask,
                               'lang_mask': language_attention_mask,
                               'token_type_ids': token_type_ids}

            h_t, language_features = self.vln_bert(**language_inputs)
            """
            h_t, language_features, language_attention_mask, token_type_ids, current_episodes, info = self.encode_instruction()

            last_dist = np.zeros(self.envs.num_envs, np.float32)
            last_ndtw = np.zeros(self.envs.num_envs, np.float32)
            for i in range(self.envs.num_envs):  # The init distance from the view point to the target
                last_dist[i] = info[i]["distance_to_goal"]
                last_ndtw[i] = info[i]["ndtw"]


            #print("h_t" , h_t.shape)
            #print("language_features" , language_features.shape)

            # Init the logs
            rewards = []
            hidden_states = []
            policy_log_probs = []
            masks = []
            entropys = []
            ml_loss = 0.

            hidden_states = []
            ended = np.array([False] * self.envs.num_envs)

            #self.dones_mask = np.array([False for _ in range(self.envs.num_envs)])

            for t in range(self.config.R2R_MODEL.max_step):
                actions, h_t, logit, candidates_mask = self.predict_next_step(h_t, language_features, language_attention_mask, token_type_ids)

                """
                env = None
                curr_elevation = None
                input_a_t, candidate_feat, candidate_leng, candidates_mask = self.get_input_feat(env, curr_elevation)

                #if (t >= 1) or (args.vlnbert=='prevalent'):
                language_features = torch.cat((h_t.unsqueeze(1), language_features[:,1:,:]), dim=1)

                visual_temp_mask = candidates_mask
                #(utils.length2mask(candidate_leng) == 0).long()
                visual_attention_mask = torch.cat((language_attention_mask, visual_temp_mask), dim=-1)

                self.vln_bert.vln_bert.config.directions = max(candidate_leng)
                ''' Visual BERT '''
                visual_inputs = {'mode':              'visual',
                                'sentence':           language_features,
                                'attention_mask':     visual_attention_mask,
                                'lang_mask':          language_attention_mask,
                                'vis_mask':           visual_temp_mask,
                                'token_type_ids':     token_type_ids,
                                'action_feats':       input_a_t,
                                # 'pano_feats':         f_t,
                                'cand_feats':         candidate_feat}
                h_t, logit = self.vln_bert(**visual_inputs)
                """

                hidden_states.append(h_t)

                candidate_mask = candidates_mask == 0

                logit.masked_fill_(candidate_mask, -float('inf'))
                # Supervised training

                target = np.zeros((self.envs.num_envs), np.longlong)
                for i in range(self.envs.num_envs):
                    target[i] = self.observations[i][expert_uuid]

                target = torch.from_numpy(target).to(self.device)

                ml_loss += self.criterion(logit, target)


                actions = torch.tensor(torch.reshape(target, (target.shape[0], 1)), device=self.device)

                #print(count , self.envs.current_episodes()[0].scene_id , self.envs.current_episodes()[0].episode_id , actions , dones)
                #print(",", actions[0].item())
                """
                for index in range(len(actions)):
                    if self.dones_mask[index] == True:
                        actions[index][0] = 0
                """
                outputs = self.envs.step([a[0].item() for a in actions])

                self.observations, reward, dones, info = [list(x) for x in zip(*outputs)]
                """
                for index in range(len(dones)):
                    if dones[index]:
                        self.dones_mask[index] = True
                """
                """
                for i in range(len(dones)):
                    logger.info(
                        f"rollout13: ,{currentframe().f_back.f_lineno},{self.log_with_episode}")
                    if not dones[i]:
                        self.set_curr_elevation(i, target[i].item())
                    else:
                        self.reset_curr_elevation(i)
                """

                log_content = ""
                for i in range(self.envs.num_envs):
                    log_content += self.envs.current_episodes()[i].scene_id.split("/")[-1] + " " + str(self.envs.current_episodes()[i].episode_id) + " " +  str(actions[i].item()) + " " + str(dones[i]) + " "
                self.log_with_episode = log_content
                logger.info(f"episode: {log_content}")


                # Determine next model inputs
                if self.feedback == 'teacher':
                    a_t = target  # teacher forcing
                elif self.feedback == 'argmax':
                    _, a_t = logit.max(1)  # student forcing - argmax
                    a_t = a_t.detach()
                    log_probs = F.log_softmax(logit,
                                              1)  # Calculate the log_prob here
                    policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(
                        1)))  # Gather the log_prob for each batch
                elif self.feedback == 'sample':
                    probs = F.softmax(logit,
                                      1)  # sampling an action from model
                    c = torch.distributions.Categorical(probs)
                    self.logs['entropy'].append(
                        c.entropy().sum().item())  # For log
                    #logger.info(f"entropy: {c.entropy().sum().item()}")
                    entropys.append(c.entropy())  # For optimization
                    a_t = c.sample().detach()
                    policy_log_probs.append(c.log_prob(a_t))
                else:
                    print(self.feedback)
                    sys.exit('Invalid feedback option')

                cpu_a_t = a_t.cpu().numpy()
                for i, next_id in enumerate(cpu_a_t):
                    if dones[i] or next_id == args.ignoreid or ended[i]:
                        cpu_a_t[i] = -1


                if train_rl:
                    # Calculate the mask and reward
                    dist = np.zeros(self.envs.num_envs, np.float32)
                    ndtw_score = np.zeros(self.envs.num_envs, np.float32)
                    reward = np.zeros(self.envs.num_envs, np.float32)
                    mask = np.ones(self.envs.num_envs, np.float32)
                    for i in range(self.envs.num_envs):
                        dist[i] = info[i]["distance_to_goal"]
                        #path_act = [vp[0] for vp in traj[i]['path']]
                        ndtw_score[i] = info[i]["ndtw"]

                        if ended[i]:
                            reward[i] = 0.0
                            mask[i] = 0.0
                        else:
                            action_idx = cpu_a_t[i]
                            # Target reward
                            if action_idx == -1:                              # If the action now is end
                                if dist[i] < 3.0:                             # Correct
                                    #reward[i] = 2.0 + ndtw_score[i] * 2.0
                                    reward[i] = 2.0
                                else:                                         # Incorrect
                                    reward[i] = -2.0
                            else:                                             # The action is not end
                                # Path fidelity rewards (distance & nDTW)
                                reward[i] = - (dist[i] - last_dist[i])
                                ndtw_reward = ndtw_score[i] - last_ndtw[i]
                                if reward[i] > 0.0:                           # Quantification
                                    reward[i] = 1.0 + ndtw_reward
                                elif reward[i] < 0.0:
                                    reward[i] = -1.0 + ndtw_reward
                                else:
                                    pass
                                    #raise NameError("The action doesn't change the move")
                                    #print("The action doesn't change the move")

                                # Miss the target penalty
                                if (last_dist[i] <= 1.0) and (dist[i]-last_dist[i] > 0.0):
                                    reward[i] -= (1.0 - last_dist[i]) * 2.0
                    rewards.append(reward)
                    masks.append(mask)
                    last_dist[:] = dist
                    last_ndtw[:] = ndtw_score
                # Update the finished actions
                # -1 means ended or ignored (already ended)
                ended[:] = np.logical_or(ended, (cpu_a_t == -1))

                # Early exit if all ended
                if ended.all():
                    break
                """
                if self.dones_mask.all():
                    break
                """

            if train_rl:
                # Last action in A2C
                env = 0
                curr_elevation = 0
                input_a_t, candidate_feat, candidate_leng, candidates_mask = self.get_input_feat(env, curr_elevation)

                language_features = torch.cat((h_t.unsqueeze(1), language_features[:,1:,:]), dim=1)

                visual_temp_mask = candidates_mask
                visual_attention_mask = torch.cat((language_attention_mask, visual_temp_mask), dim=-1)

                self.vln_bert.vln_bert.config.directions = max(candidate_leng)
                ''' Visual BERT '''
                visual_inputs = {'mode':              'visual',
                                'sentence':           language_features,
                                'attention_mask':     visual_attention_mask,
                                'lang_mask':          language_attention_mask,
                                'vis_mask':           visual_temp_mask,
                                'token_type_ids':     token_type_ids,
                                'action_feats':       input_a_t,
                                # 'pano_feats':         f_t,
                                'cand_feats':         candidate_feat}
                last_h_, _ = self.vln_bert(**visual_inputs)

                rl_loss = 0.

                # NOW, A2C!!!
                # Calculate the final discounted reward
                last_value__ = self.critic(last_h_).detach()        # The value esti of the last state, remove the grad for safety
                discount_reward = np.zeros(self.envs.num_envs, np.float32)  # The inital reward is zero
                for i in range(self.envs.num_envs):
                    if not ended[i]:        # If the action is not ended, use the value function as the last reward
                        discount_reward[i] = last_value__[i]

                length = len(rewards)
                total = 0
                for t in range(length-1, -1, -1):
                    discount_reward = discount_reward * args.gamma + rewards[t]  # If it ended, the reward will be 0
                    mask_ = Variable(torch.from_numpy(masks[t]), requires_grad=False).cuda(self.device)
                    clip_reward = discount_reward.copy()
                    r_ = Variable(torch.from_numpy(clip_reward), requires_grad=False).cuda(self.device)
                    v_ = self.critic(hidden_states[t])
                    a_ = (r_ - v_).detach()

                    rl_loss += (-policy_log_probs[t] * a_ * mask_).sum()
                    rl_loss += (((r_ - v_) ** 2) * mask_).sum() * 0.5  # 1/2 L2 loss
                    if self.feedback == 'sample':
                        rl_loss += (- 0.01 * entropys[t] * mask_).sum()
                    self.logs['critic_loss'].append((((r_ - v_) ** 2) * mask_).sum().item())
                    #logger.info(f"critic_loss: {(((r_ - v_) ** 2) * mask_).sum().item()}")

                    total = total + np.sum(masks[t])
                self.logs['total'].append(total)
                logger.info(f"total: {total}")

                # Normalize the loss function
                if args.normalize_loss == 'total':
                    rl_loss /= total
                elif args.normalize_loss == 'batch':
                    rl_loss /= self.envs.num_envs
                else:
                    assert args.normalize_loss == 'none'

                self.loss += rl_loss
                self.logs['RL_loss'].append(rl_loss.item())
                logger.info(f"RL_loss: {rl_loss.item()}")

            if train_ml is not None:
                self.loss += ml_loss * train_ml / self.envs.num_envs
                self.logs['IL_loss'].append((ml_loss * train_ml / self.envs.num_envs).item())
                logger.info(f"IL_loss: {(ml_loss * train_ml / self.envs.num_envs).item()}")

            if type(self.loss) is int:  # For safety, it will be activated if no losses are added
                self.losses.append(0.)
            else:
                self.losses.append(self.loss.item() / self.config.R2R_MODEL.episode_len)  # This argument is useless.




    def get_input_feat(self, env, curr_elevation):
        input_a_t = np.zeros((self.envs.num_envs , self.config.R2R_MODEL.angle_feat_size), np.float32)
        candidates, candidates_mask, candidate_leng = self.make_candidates()

        heading = 0
        elevation = 0
        for i in range(self.envs.num_envs):
            input_a_t[i,:] = calculate_angle_feature(heading, elevation)
        #input_a_t = input_a_t.reshape(1 , input_a_t.shape[0])
        input_a_t = torch.from_numpy(input_a_t).cuda(self.device)
        candidate_feat = candidates.cuda(self.device)

        return input_a_t, candidate_feat, candidate_leng, candidates_mask

    def _candidate_variable(self, obs):
        candidate_leng = [6]  # +1 is for the end
        candidate_feat = np.zeros((len(obs), max(candidate_leng), self.config.R2R_MODEL.feature_size + self.config.R2R_MODEL.angle_feat_size), dtype=np.float32)
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, cc in enumerate(ob['candidate']):
                candidate_feat[i, j, :] = cc['feature']

        return torch.from_numpy(candidate_feat).cuda(), candidate_leng

    def encode_instr(self, instr):
        tokenizer_class = BertTokenizer
        tokenizer = tokenizer_class.from_pretrained('bert-base-uncased')
        ''' BERT tokenizer '''
        instr_tokens = tokenizer.tokenize(instr)
        padded_instr_tokens, num_words = self.pad_instr_tokens(instr_tokens, self.config.R2R_MODEL.max_action)
        instr_encoding = tokenizer.convert_tokens_to_ids(padded_instr_tokens)
        return instr_encoding


    def pad_instr_tokens(self, instr_tokens, maxlength=20):

        if len(instr_tokens) <= 2:  # assert len(raw_instr_tokens) > 2
            return None

        if len(instr_tokens) > maxlength - 2:  # -2 for [CLS] and [SEP]
            instr_tokens = instr_tokens[:(maxlength - 2)]

        instr_tokens = ['[CLS]'] + instr_tokens + ['[SEP]']
        num_words = len(instr_tokens)  # - 1  # include [SEP]
        instr_tokens += ['[PAD]'] * (maxlength - len(instr_tokens))

        assert len(instr_tokens) == maxlength

        return instr_tokens, num_words

    def _sort_batch(self, instr_encoding):
        #seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_tensor = np.array(instr_encoding)
        seq_lengths = np.argmax(seq_tensor == utils.padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]

        seq_tensor = torch.from_numpy(seq_tensor)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Sort sequences by lengths
        seq_lengths, perm_idx = seq_lengths.sort(0, True)  # True -> descending
        sorted_tensor = seq_tensor[perm_idx]
        mask = (sorted_tensor != utils.padding_idx)

        token_type_ids = torch.zeros_like(mask)

        return Variable(sorted_tensor, requires_grad=False).long().to(self.device), \
               mask.long().to(self.device), token_type_ids.long().to(self.device), \
               list(seq_lengths), list(perm_idx)

    def make_candidates(self):
        # elevation 0:look up, 1:look forward, 2:look down
        action_num = self.config.R2R_MODEL.action_num
        image_feature_size = self.config.R2R_MODEL.image_feature_size
        angle_feature_size = self.config.R2R_MODEL.angle_feature_size
        self.candidates_mask = torch.tensor([[1 for i in range(action_num)] for _ in range(self.envs.num_envs)]).to(self.device)
        self.candidate_leng = np.array([[action_num] for _ in range(self.envs.num_envs)])
        self.candidates = torch.ones(
            self.envs.num_envs,
            action_num,
            image_feature_size + angle_feature_size,
           device=self.device
        )

        curr_elevation_flag_up = [[False] for _ in range(self.envs.num_envs)]
        curr_elevation_flag_down = [[False] for _ in range(self.envs.num_envs)]
        curr_elevation_flag = [[False] for _ in range(self.envs.num_envs)]
        """
        for i in range(self.envs.num_envs):
            if self.curr_elevation[i] == 0:
                curr_elevation_flag_up[i] = True

            elif self.curr_elevation[i] == 2:
                curr_elevation_flag_down[i] = True
        logger.info(f"make_candidates(3),{currentframe().f_back.f_lineno},{self.log_with_episode}")
        """

        self.candidates[:, 0, :] = torch.ones(image_feature_size + angle_feature_size) * (-1)
        self.candidates_mask[:, 0] = 0
        self.candidate_leng[:] -= 1
        self.candidates[:, 1, :] = self.get_forward_image(curr_elevation_flag)
        self.candidates[:, 2, :] = self.get_left_image(curr_elevation_flag)
        self.candidates[:, 3, :] = self.get_right_image(curr_elevation_flag)
        """
        self.candidates[:, 4, :] = self.get_up_image(curr_elevation_flag_up)
        self.candidates[:, 5, :] = self.get_down_image(curr_elevation_flag_down)
        """
        """
        for i in range(self.envs.num_envs):
            if self.curr_elevation[i] == 0:
                self.candidates[i, 4, :] = torch.ones(image_feature_size + angle_feature_size)*(-1)
                self.candidates_mask[i,4] = 0
                self.candidate_leng[i] -= 1

            elif self.curr_elevation[i] == 2:
                self.candidates[i, 5, :] = torch.ones(image_feature_size + angle_feature_size)*(-1)
                self.candidates_mask[i,5] = 0
                self.candidate_leng[i] -= 1
        """

        return self.candidates, self.candidates_mask, self.candidate_leng

    def go_step(self, index, heading, elevation, up_down_flag, need_return = False):
        #up_down_flag = [[False], [True], [False]]
        index_arr = [[index] for _ in range(self.envs.num_envs)]
        """
        for item_index in range(len(up_down_flag)):
            if up_down_flag[item_index][0] == True and index >=4 :
                index_arr[item_index][0] -= 2
        """
        actions = torch.tensor(index_arr, device=self.device)

        #print(actions)
        if index == 1:
            observations = self.observations
        else:
            """
            for index in range(len(actions)):
                if self.dones_mask[index] == True:
                    actions[index][0] = 0
            """
            outputs = self.envs.step([a[0].item() for a in actions])
            observations, _, dones, _ = [list(x) for x in zip(*outputs)]
            #print("get_done", self.envs.call_at(0, "get_done", {"observations": self.observations}))


        # to-delete
        #img = observations[0]['rgb']

        #plt.imshow(img)
        #plt.show()

        if need_return:
            #print(self.images[index].shape)
            angle_feature = calculate_angle_feature(heading, elevation)
            angle_feature = torch.from_numpy(angle_feature)
            angle_feature = angle_feature.reshape(1 , angle_feature.shape[0])
            angle_feature = angle_feature.to(self.device)
            feature = torch.ones((self.envs.num_envs, self.config.R2R_MODEL.image_feature_size + self.config.R2R_MODEL.angle_feature_size))*(-1)
            #for i in range(self.envs.num_envs):

            image_feature = self.get_image_features(observations)
            #image_feature = self.get_image_features(observations[i]["rgb"])
            feature = torch.cat((image_feature, torch.cat((angle_feature , angle_feature, angle_feature) , dim = 0)), dim=1)
            return feature

    def go_forward(self, up_down_flag, need_return = False):
        heading = 0
        elevation = 0
        index = 1
        feature = self.go_step(index, heading, elevation, up_down_flag, need_return)
        return feature

    def turn_left(self, up_down_flag, need_return = False):
        heading = 0
        elevation = -math.pi/6
        index = 2
        feature = self.go_step(index, heading, elevation, up_down_flag, need_return)
        return feature

    def turn_right(self, up_down_flag, need_return = False):
        heading = 0
        elevation = math.pi/6
        index = 3
        feature = self.go_step(index, heading, elevation, up_down_flag, need_return)
        return feature

    def look_up(self, up_down_flag,need_return = False):
        heading = math.pi/6
        elevation = 0
        index = 4
        feature = self.go_step(index, heading, elevation, up_down_flag, need_return)
        return feature

    def look_down(self, up_down_flag, need_return = False):
        heading = -math.pi/6
        elevation = 0
        index = 5
        feature = self.go_step(index, heading, elevation, up_down_flag, need_return)
        return feature


    def get_forward_image(self, up_down_flag):
        #print("get_forward_image")
        image = self.go_forward(up_down_flag, need_return = True)
        """
        for i in range(6):
            self.turn_left()
        self.go_forward()
        for i in range(6):
            self.turn_left()
        """
        return image

    def get_left_image(self, up_down_flag):
        #print("get_left_image")
        image = self.turn_left(up_down_flag, need_return = True)
        self.turn_right(up_down_flag)
        return image

    def get_right_image(self, up_down_flag):
        #print("get_right_image")
        image = self.turn_right(up_down_flag, need_return = True)
        self.turn_left(up_down_flag)
        return image

    def get_up_image(self, up_down_flag):
        #print("get_up_image")
        image = self.look_up(up_down_flag, need_return = True)
        self.look_down(up_down_flag)
        return image

    def get_down_image(self, up_down_flag):
        #print("get_down_image")
        image = self.look_down(up_down_flag, need_return = True)
        self.look_up(up_down_flag)
        return image

    def get_image_features(self , image):

        #plt.imshow(image)
        #plt.show()

        #image = torch.from_numpy(image).float()
        #image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
        #image = image.to(self.device)



        #self.model.cuda(device)


        self.obs_transforms = get_active_obs_transforms(self.config)
        image = batch_obs(image, self.device)
        image = apply_obs_transforms_batch(image, self.obs_transforms)

        feature = self.model(image['rgb'])
        feature = feature.to(self.device)

        return feature

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)

        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            if args.loadOptim:
                optimizer.load_state_dict(states[name]['optimizer'])
        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            recover_state(*param)
        return states['vln_bert']['epoch'] - 1


