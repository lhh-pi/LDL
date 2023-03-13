import matplotlib.pyplot as plt

loss = {'l_g_pix': [],
        'l_g_artifacts': [],
        'l_g_bp': [],
        'l_g_percep': [],
        'l_g_gan': [],
        'l_g_total': [],
        'l_d_real': [],
        'l_d_fake': []}
loss_type = ['l_g_pix', 'l_g_artifacts', 'l_g_bp', 'l_g_percep', 'l_g_gan', 'l_g_total', 'l_d_real', 'l_d_fake']
len_num = 10  # 数据长度

# log_path = '../experiments/train_MapSR_LDL_RA_BP_pretrain_archived_20230312_141830/train_train_MapSR_LDL_RA_BP_pretrain_20230311_205258.log'
# log_path = '../experiments/train_MapSR_LDL_RA_BP_pretrain/train_train_MapSR_LDL_RA_BP_pretrain_20230313_113210.log'
log_path = '../experiments/train_mapsr_LDL_RA_BP_pretrain_gan_5e-3_percep_1e-2_lr_1e-4/train_train_mapsr_LDL_RA_BP_pretrain_gan_5e-3_percep_1e-2_lr_1e-4_20230313_133442.log'

with open(log_path, 'r') as f:
    line = f.readline()
    while line:
        line = f.readline()
        index = line.find(loss_type[0])
        if index > 0:
            # l_g_pix
            i = 0
            index = line.find(loss_type[i]) + len(loss_type[i]) + 2
            loss[loss_type[i]].append(float(line[index: index + len_num]))

            # l_g_artifacts
            i = 1
            index = line.find(loss_type[i]) + len(loss_type[i]) + 2
            loss[loss_type[i]].append(float(line[index: index + len_num]))

            # l_g_bp
            i = 2
            index = line.find(loss_type[i]) + len(loss_type[i]) + 2
            loss[loss_type[i]].append(float(line[index: index + len_num]))

            # l_g_percep
            i = 3
            index = line.find(loss_type[i]) + len(loss_type[i]) + 2
            loss[loss_type[i]].append(float(line[index: index + len_num]))

            # l_g_gan
            i = 4
            index = line.find(loss_type[i]) + len(loss_type[i]) + 2
            loss[loss_type[i]].append(float(line[index: index + len_num]))

            # l_g_total
            i = 5
            loss[loss_type[i]].append(sum([loss[loss_type[j]][-1] for j in range(i)]))

            # l_d_real
            i = 6
            index = line.find(loss_type[i]) + len(loss_type[i]) + 2
            loss[loss_type[i]].append(float(line[index: index + len_num]))

            # l_g_fake
            i = 7
            index = line.find(loss_type[i]) + len(loss_type[i]) + 2
            loss[loss_type[i]].append(float(line[index: index + len_num]))

# print(loss)

freq = 100  # log输出间隔
x = [freq * (i + 1) for i in range(len(loss[loss_type[0]]))]  # x轴

# ax = [None for _ in range(len(loss))]
fig, axs = plt.subplots(nrows=4, ncols=2, sharex=True, figsize=(8, 12))

# 将子图数组展平为一维数组，以便使用 for 循环遍历每个子图
axs = axs.flatten()

# for i in range(8):
#     axs[i].set_ylabel(str(loss_type[i]))
#     axs[i].plot(x, loss[loss_type[i]], color='green')

# 在每个子图中绘制数据并设置标题
for i, ax in enumerate(axs):
    y = loss[loss_type[i]]
    # print(loss_type[i], ': ', y)
    ax.plot(x, y)
    ax.set_title(f"Plot {loss_type[i]}")

# 设置共享的 x 和 y 轴标签
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("iter")
plt.ylabel("loss")


plt.show()  # 显示图表
