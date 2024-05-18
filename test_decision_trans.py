import numpy as np

# 指定npz文件的路径
# file_path = '/home/maoyc/offline_multitask/collected_data/walker_run-td3-medium/data/episode_000950_1000.npz'

# 指定现有npz文件的路径
existing_file_path = '/home/maoyc/offline_multitask/collected_data/walker_run-td3-medium/data/episode_000950_1000.npz'

# 读取现有npz文件
data = np.load(existing_file_path)

# 将现有数据转换为字典
existing_data = {name: data[name] for name in data.files}

for da in existing_data:
    print(da)

# 新增数据
new_data = {
    'new_array1': np.array([10, 20, 30]),
    'new_array2': np.array([[1, 2], [3, 4]])
}

# 合并现有数据和新增数据
combined_data = {**existing_data, **new_data}

# 指定保存合并后数据的npz文件路径
# new_file_path = 'updated_file.npz'

# 保存合并后的数据到新的npz文件
# np.savez(new_file_path, **combined_data)

# 关闭现有npz文件
data.close()

# print(f"新的npz文件已保存到: {new_file_path}")
