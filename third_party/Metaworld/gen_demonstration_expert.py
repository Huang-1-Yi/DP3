# bash scripts/metaworld/gen_demonstration_expert.sh reach 5
import argparse
import os
import zarr
import numpy as np
from diffusion_policy_3d.env import MetaWorldEnv
from termcolor import cprint
import copy
import imageio
from metaworld.policies import *
# 被注释掉的故障处理代码，用于调试时定位程序崩溃位置
# import faulthandler
# faulthandler.enable()

# 生成随机种子（虽然代码后续未使用）
seed = np.random.randint(0, 100)


import matplotlib.pyplot as plt
import matplotlib.animation as animation


# # 创建一个画布来显示图像
# fig, ax = plt.subplots()
# img = ax.imshow(np.zeros((480, 640, 3)), animated=True)  # 初始化为空图像
# def update_frame(frame):
#     # 这里 frame 是图像数据
#     img.set_array(frame)  # 更新图像数据
#     return [img]



def load_mw_policy(task_name):
	"""
	策略加载函数：
		特殊处理 'peg-insert-side' 任务
		其他任务通过字符串处理构造类名（如 'reach' 转换为 SawyerReachV2Policy）
		使用 eval() 动态实例化策略类
	"""
	if task_name == 'peg-insert-side':
		agent = SawyerPegInsertionSideV2Policy()
	else:
		task_name = task_name.split('-')
		task_name = [s.capitalize() for s in task_name]
		task_name = "Sawyer" + "".join(task_name) + "V2Policy"
		agent = eval(task_name)()
	return agent

def main(args):
	env_name = args.env_name

	# 处理数据存储路径：
		# 构建 Zarr 文件路径
		# 检查路径是否存在并提示覆盖
		# 强制设置为自动覆盖（实际代码中 user_input 硬编码为 'y'）
		# 创建存储目录
	save_dir = os.path.join(args.root_dir, 'metaworld_'+args.env_name+'_expert.zarr')
	if os.path.exists(save_dir):
		cprint('Data already exists at {}'.format(save_dir), 'red')
		cprint("If you want to overwrite, delete the existing directory first.", "red")
		cprint("Do you want to overwrite? (y/n)", "red")
		user_input = 'y'
		if user_input == 'y':
			cprint('Overwriting {}'.format(save_dir), 'red')
			os.system('rm -rf {}'.format(save_dir))
		else:
			cprint('Exiting', 'red')
			return
	os.makedirs(save_dir, exist_ok=True)

	# 初始化 MetaWorld 环境：
		# 指定任务名称

		# 使用 CUDA 加速（device="cuda:0"）

		# 启用点云裁剪（use_point_crop=True）
	e = MetaWorldEnv(env_name, device="cuda:0", use_point_crop=True)
	# 获取并打印要收集的 episode 数量
	num_episodes = args.num_episodes
	cprint(f"Number of episodes : {num_episodes}", "yellow")
	
	# 初始化数据存储列表：
	total_count = 0
	img_arrays = []			# 图像观察
	# # 图像实时显示
	# animation.FuncAnimation(fig, update_frame, frames=obs_img, interval=50, blit=True)  # 50ms 更新一次
	# plt.show()
	point_cloud_arrays = []	# 点云数据
	depth_arrays = []		# 深度图
	state_arrays = []		# 机器人状态
	full_state_arrays = []	# 完整环境状态
	action_arrays = []		# 专家动作
	episode_ends_arrays = []# 记录每个 episode 的结束索引
    
	# 初始化 episode 计数器并加载专家策略
	episode_idx = 0
	

	mw_policy = load_mw_policy(env_name)
	
	# 开始 episode 循环：loop over episodes
	while episode_idx < num_episodes:
		# 重置环境并获取初始状态
		# 获取初始视觉观察
		# 初始化 episode 相关变量（奖励、成功标志等）
		raw_state = e.reset()['full_state']
		obs_dict = e.get_visual_obs()
		done = False
		ep_reward = 0.
		ep_success = False
		ep_success_times = 0
		
		# 为当前 episode 初始化临时存储列表
		img_arrays_sub = []
		point_cloud_arrays_sub = []
		depth_arrays_sub = []
		state_arrays_sub = []
		full_state_arrays_sub = []
		action_arrays_sub = []
		total_count_sub = 0

		# 开始时间步循环，收集各类观察数据：

		while not done:
			total_count_sub += 1

			obs_img = obs_dict['image']					# 	图像
			obs_robot_state = obs_dict['agent_pos']		# 	机器人位置状态
			obs_point_cloud = obs_dict['point_cloud']	# 	点云
			obs_depth = obs_dict['depth']				# 	深度图		
   
			# 将当前观察数据添加到临时列表
			img_arrays_sub.append(obs_img)
			point_cloud_arrays_sub.append(obs_point_cloud)
			depth_arrays_sub.append(obs_depth)
			state_arrays_sub.append(obs_robot_state)
			full_state_arrays_sub.append(raw_state)
			
			# 使用专家策略生成动作，并存储动作数据
			action = mw_policy.get_action(raw_state)
			action_arrays_sub.append(action)

			# 执行动作并更新环境状态：
			# 	累积奖励
			# 	更新成功标志
			# 	遇到终止条件时退出循环
			obs_dict, reward, done, info = e.step(action)
			raw_state = obs_dict['full_state']
			ep_reward += reward
   
			ep_success = ep_success or info['success']
			ep_success_times += info['success']
   
			if done:
				break
		
		# 筛选成功 episode
		#     必须至少成功一次且成功次数 ≥5 次（确保轨迹质量）
		# 保存合格 episode 的数据：
		# 	记录 episode 结束索引
		# 	将临时数据合并到主列表
		# 	更新成功 episode 计数器
		if not ep_success or ep_success_times < 5:
			cprint(f'Episode: {episode_idx} failed with reward {ep_reward} and success times {ep_success_times}', 'red')
			continue
		else:
			total_count += total_count_sub
			episode_ends_arrays.append(copy.deepcopy(total_count)) # the index of the last step of the episode    
			img_arrays.extend(copy.deepcopy(img_arrays_sub))
			point_cloud_arrays.extend(copy.deepcopy(point_cloud_arrays_sub))
			depth_arrays.extend(copy.deepcopy(depth_arrays_sub))
			state_arrays.extend(copy.deepcopy(state_arrays_sub))
			action_arrays.extend(copy.deepcopy(action_arrays_sub))
			full_state_arrays.extend(copy.deepcopy(full_state_arrays_sub))
			cprint('Episode: {}, Reward: {}, Success Times: {}'.format(episode_idx, ep_reward, ep_success_times), 'green')
			episode_idx += 1
	

	# save data
 	###############################
	# 创建 Zarr 存储结构：
	# 	data 组存储实际数据
	# 	meta 组存储元数据
    # # create zarr file
	zarr_root = zarr.group(save_dir)
	zarr_data = zarr_root.create_group('data')
	zarr_meta = zarr_root.create_group('meta')
	# save img, state, action arrays into data, and episode ends arrays into meta
	# 数据预处理：
	# 	将列表转换为 numpy 数组
	# 	调整图像通道顺序（从 CxHxW 转为 HxWxC）
	img_arrays = np.stack(img_arrays, axis=0)
	if img_arrays.shape[1] == 3: # make channel last
		img_arrays = np.transpose(img_arrays, (0,2,3,1))
	state_arrays = np.stack(state_arrays, axis=0)
	full_state_arrays = np.stack(full_state_arrays, axis=0)
	point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
	depth_arrays = np.stack(depth_arrays, axis=0)
	action_arrays = np.stack(action_arrays, axis=0)
	episode_ends_arrays = np.array(episode_ends_arrays)

	# 配置 Zarr 存储参数：
	# 	使用 Zstandard 压缩算法
	# 	定义合理的数据块大小（chunk）以优化存储效率
	compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
	img_chunk_size = (100, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3])
	state_chunk_size = (100, state_arrays.shape[1])
	full_state_chunk_size = (100, full_state_arrays.shape[1])
	point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
	depth_chunk_size = (100, depth_arrays.shape[1], depth_arrays.shape[2])
	action_chunk_size = (100, action_arrays.shape[1])
	zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
	zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
	zarr_data.create_dataset('full_state', data=full_state_arrays, chunks=full_state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
	zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
	zarr_data.create_dataset('depth', data=depth_arrays, chunks=depth_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
	zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
	zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)

	# 输出数据统计信息并确认保存位置。
	cprint(f'-'*50, 'cyan')
	# print shape 打印各数据的形状和数值范围
	cprint(f'img shape: {img_arrays.shape}, range: [{np.min(img_arrays)}, {np.max(img_arrays)}]', 'green')
	cprint(f'point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
	cprint(f'depth shape: {depth_arrays.shape}, range: [{np.min(depth_arrays)}, {np.max(depth_arrays)}]', 'green')
	cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
	cprint(f'full_state shape: {full_state_arrays.shape}, range: [{np.min(full_state_arrays)}, {np.max(full_state_arrays)}]', 'green')
	cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
	cprint(f'Saved zarr file to {save_dir}', 'green')

	# clean up
	# 清理内存，删除不再需要的大数组
	del img_arrays, state_arrays, point_cloud_arrays, action_arrays, episode_ends_arrays
	del zarr_root, zarr_data, zarr_meta
	del e


 
if __name__ == "__main__":
    
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', type=str, default='basketball')
	parser.add_argument('--num_episodes', type=int, default=10)
	parser.add_argument('--root_dir', type=str, default="../../3D-Diffusion-Policy/data/" )

	args = parser.parse_args()
	main(args)
