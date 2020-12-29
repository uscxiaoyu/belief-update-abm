# coding=utf-8

import time
import datetime
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import multiprocessing

NUM_CORNS = multiprocessing.cpu_count()  # cpu核数


def generate_normal_value(mu, sigma, m, n):
    """生成(m,n)之间符合正态分布的随机数"""
    while True:
        a = np.random.normal(mu, sigma)
        if m <= a <= n:
            return a


def generate_lognormal_values(t_mean, size=10000, tau=0.1):
    mu, sigma = 0, 1
    i = 0
    while True:
        a = np.random.lognormal(mean=mu, sigma=sigma, size=size)
        if np.mean(a) / t_mean > 1.01:
            mu -= tau * np.random.random()
        elif np.mean(a) / t_mean < 0.99:
            mu += tau * np.random.random()
        else:
            break
        i += 1

    j = 0
    while True:
        large_a = a[a > np.mean(a)]
        small_a = a[a <= np.mean(a)]
        ratio = np.std(a) / np.mean(a)
        r = np.random.random()
        if ratio > 1.01:
            cum_a = np.sum(large_a) * tau * r
            large_a = large_a * (1 - tau * r)
            small_a += np.array([cum_a / len(small_a) for i in range(len(small_a))])
        elif ratio < 0.99:
            cum_a = np.sum(small_a) * tau * r
            small_a = small_a * (1 - tau * r)
            large_a += np.array([cum_a / len(large_a) for i in range(len(large_a))])
        else:
            break
        a = np.concatenate((small_a, large_a), axis=None)
        j += 1

    np.random.shuffle(a)
    # print(f'调整方差: 执行{j}轮结束！')
    return a


def adjust_ratio(x, r_list, real_std_r):
    """
    Goal: adjust the st.d of r_list to real_std_r while keep the mean fixed.
    Function: decrease/increase the value of r above the mean, then increase/decrease the value of r below/equal to the mean. Keep the ends meet.
    Input: x---ratio, r_list--the r list from agents, real_std_r---the target st.d of r
    Output: the difference between the st.d of the adjusted r_list and the real_std_r
    """
    mean_r = np.mean(r_list)
    large_r_list = [r for r in r_list if r > mean_r]
    small_r_list = [r for r in r_list if r <= mean_r]

    ch_r = sum([x * r for r in large_r_list]) / len(small_r_list)  # 减小比例
    large_r_list = [(1 - x) * r for r in large_r_list]  # 减小之后的列表
    small_r_list = [r + ch_r for r in small_r_list]  # 增大之后的列表

    return np.std(small_r_list + large_r_list) - real_std_r


def logit(x, alpha=0.5):
    return 1 / (1 + np.exp(-x * alpha))


class AgentDiffusionModel:
    def __init__(
        self,
        delta,  # the effect of Advertisement
        mean_q,
        perc_disp_ratio,
        lower_d=-2,
        upper_d=5,
        G=nx.random_graphs.barabasi_albert_graph(1000, 3),
        seed_type="social hub",
        omega=2,  # the relative effect of negative WOM over positive WOM to 2.
        travel_degree=2,  # the decay of effect
        u_adopt=1,  # utility if agent adopt a good innovation
        u_reject=1,  # utility if agent adopt a bad innovation
    ):
        """
        G: the network instance on which the abms build
        seed_type: social hubs, experts, randomly designated
        state: 0---undecided, 1---adopt, -1---dissatified, -2---rejected
        """
        if G.is_directed():
            self.G = deepcopy(G)
        else:
            self.G = deepcopy(G.to_directed())

        self.delta = delta  # 广告影响，虽然paper描述为mean，但没看到具体分布
        self.mean_q = mean_q
        self.omega = omega  # negative factor
        # negative WOM only travels 2 degrees of separation
        self.travel_degree = travel_degree
        q_array = generate_lognormal_values(t_mean=self.mean_q, size=self.G.number_of_nodes())
        # np.random.lognormal()
        # 对创新为好创新的置信度，为正时表示相信其为好创新；为负时表示其为不好的创新
        self.innovGood_degree_array = np.random.randint(lower_d, upper_d, size=self.G.number_of_nodes())
        self.u_adopt = u_adopt
        self.u_reject = u_reject
        # 以下节点属性在仿真过程中保持不变
        for i, x in enumerate(self.G.nodes()):
            self.G.nodes[x]["delta"] = self.delta
            self.G.nodes[x]["q"] = q_array[i]
            # 在创新实际上为好的创新前提下，agent认为其下一个接收到是正面口碑的概率
            self.G.nodes[x]["p_pos_innovGood"] = 0.5 * np.random.random() + 0.5
            # 在创新实际上为坏的创新前提下，agent认为其下一个接收到是负面口碑的概率
            self.G.nodes[x]["p_neg_innovBad"] = 0.5 * np.random.random() + 0.5
            # 认知能力上限，即可以处理的信息的数量
            self.G.nodes[x]["cognitive_limit"] = np.random.randint(1, 5)
            self.G.nodes[x]["predecessor"] = list(self.G.predecessors(x))
            self.G.nodes[x]["successor"] = list(self.G.successors(x))
            # agent使用创新后是否为失望
            self.G.nodes[x]["isDisappointed"] = False

        self.seed_type = seed_type
        self.num_seeds = int(perc_disp_ratio * self.G.number_of_nodes())  # 不满意节点数量
        self.seeds = self.choose_seeds()
        for i in self.seeds:
            self.G.nodes[i]["isDisappointed"] = True

    def initilize_state(self):
        """
        Goal: initialize the state of agents.
        """
        # 以下节点属性在仿真过程中动态变化
        for i, x in enumerate(self.G.nodes()):
            self.G.nodes[x]["state"] = 0  # 当前时间步所处的状态
            self.G.nodes[x]["next_state"] = 0  # 下一个时间步的状态
            # agent认为创新是好创新的先验权值
            self.G.nodes[x]["innovGood_degree"] = self.innovGood_degree_array[i]
            # A dict of whether WOM of the Agent's neighbors is process(1) or ignore(0){neighbor j : 0 or 1}
            self.G.nodes[x]["WOM_treate_table"] = {j: 0 for j in self.G.nodes[x]["predecessor"]}

    def choose_seeds(self):
        """
        Goal: identify the seeds of disappointed agents according to the strategy of selection.
        seed_type: randomly designated, revenue leader, social hub, expert, base_case
        Output: The set of seeds.
        """
        if self.seed_type == "randomly designated":  # randomly designagted
            return np.random.choice(self.G.nodes, self.num_seeds, replace=False)
        elif self.seed_type == "social hub":  # social hubs
            return sorted([i for i in self.G.nodes()], key=lambda x: len(self.G.nodes[x]["successor"]), reverse=True)[
                : self.num_seeds
            ]
        elif self.seed_type == "expert":  # experts
            return sorted([i for i in self.G.nodes()], key=lambda x: self.G.nodes[x]["q"], reverse=True)
        else:  # no seed, base case
            return []

    def process_WOM(self, i, j, state, p_seq_innovGood, p_seq_innovBad, count):
        """
        i -- agent i
        j -- agent i's neighbor j
        state -- neighbor j's WOM based on j's state; state: +1 --> PWOM; -1, -2 --> NWOM
        count -- 该时间步内agent i已经处理的WOM数
        """
        # agent i对创新为好的创新的初始信念
        p_innovGood = logit(self.G.nodes[i]["innovGood_degree"])
        p_innovBad = 1 - p_innovGood
        # 如果agent认为现实是这应该是一个好(坏)的创新，应该接收到PWOM(NWOM)的概率。
        p_pos_innovGood, p_neg_innovGood = [self.G.nodes[i]["p_pos_innovGood"], 1 - self.G.nodes[i]["p_pos_innovGood"]]
        p_neg_innovBad, p_pos_innovBad = [self.G.nodes[i]["p_neg_innovBad"], 1 - self.G.nodes[i]["p_neg_innovBad"]]

        prob_PWOM = p_innovGood * p_pos_innovGood + p_innovBad * p_neg_innovBad  # 下一条WOM为PWOM的全概率
        prob_NWOM = p_innovBad * p_neg_innovBad + p_innovGood * p_neg_innovGood  # 下一条WOM为NWOM的全概率

        # 根据邻居的状态(对应正面或负面口碑)选择不同的计算方法
        if state > 0:
            # 如果处理，则根据当前WOM的状态计算效用
            utility_process = (
                self.u_adopt
                * p_innovGood
                * p_pos_innovGood
                * p_seq_innovGood
                / (p_innovGood * p_pos_innovGood * p_seq_innovGood + p_innovBad * p_neg_innovBad * p_seq_innovBad)
            )
            # 如果忽略当前WOM，则结合下一条WOM的可能状态计算期望效用
            utility_ignore = prob_PWOM * self.u_adopt * (
                p_innovGood
                * p_pos_innovGood
                * p_pos_innovGood
                * p_seq_innovGood
                / (
                    p_innovGood * p_pos_innovGood * p_pos_innovGood * p_seq_innovGood
                    + p_innovBad * p_neg_innovBad * p_neg_innovBad * p_seq_innovBad
                )
            ) + prob_NWOM * self.u_reject * (
                p_innovBad
                * p_neg_innovBad
                * p_pos_innovBad
                * p_seq_innovBad
                / (
                    p_innovGood * p_pos_innovGood * p_neg_innovGood * p_seq_innovGood
                    + p_innovBad * p_neg_innovBad * p_pos_innovBad * p_seq_innovBad
                )
            )

            p_seq_innovGood *= p_pos_innovGood  # 更新创新为好创新下的联合概率分布，接收到的为PWOM，乘上p
            p_seq_innovBad *= p_neg_innovBad  # 更新创新为坏创新下的联合概率分布，接收到的为PWOM，乘上1-p
            if utility_process > utility_ignore:
                self.G.nodes[i]["WOM_treate_table"][j] = 1
                # 处理PWOM之后，所以好创新对应的置信度+1
                self.G.nodes[i]["innovGood_degree"] = self.G.nodes[i]["innovGood_degree"] + 1
                count += 1

        # 当前邻近的state=-1或-2，即处于未决定状态和拒绝状态的邻近都影响i的信念
        if state < 0:
            utility_process = (
                self.u_reject
                * p_innovBad
                * p_pos_innovBad
                * p_seq_innovBad
                / (p_innovGood * p_neg_innovGood * p_seq_innovGood + p_innovBad * p_pos_innovBad * p_seq_innovBad)
            )
            utility_ignore = prob_PWOM * self.u_adopt * (
                p_innovGood
                * p_neg_innovGood
                * p_pos_innovGood
                * p_seq_innovGood
                / (
                    p_innovGood * p_neg_innovGood * p_pos_innovGood * p_seq_innovGood
                    + p_innovBad * p_pos_innovBad * p_neg_innovBad * p_seq_innovBad
                )
            ) + prob_NWOM * self.u_reject * (
                p_innovBad
                * p_pos_innovBad
                * p_pos_innovBad
                * p_seq_innovBad
                / (
                    p_innovGood * p_neg_innovGood * p_neg_innovGood * p_seq_innovGood
                    + p_innovBad * p_pos_innovBad * p_pos_innovBad * p_seq_innovBad
                )
            )

            p_seq_innovGood *= p_neg_innovGood  # 更新创新为好创新下的联合概率分布，接收到的为NWOM，乘上1-p
            p_seq_innovBad *= p_pos_innovBad  # 更新创新为坏创新下的联合概率分布，接收到的为NWOM，乘上p
            if utility_process > utility_ignore:
                self.G.nodes[i]["WOM_treate_table"][j] = 1
                # 处理了负面信息，所以好创新对应的置信度-1
                self.G.nodes[i]["innovGood_degree"] = self.G.nodes[i]["innovGood_degree"] - 1
                count += 1

        return p_seq_innovGood, p_seq_innovBad, count

    def choose_WOM(self, i):
        """
        agent从邻居发送的信息中选择信息。
        """
        p_seq_innovGood = 1  # 如果agent认为这是一个好的创新，判断是否处理m+1条WOM之前，前m条WOM出现的累计概率
        p_seq_innovBad = 1  # 如果agent认为这是一个坏的创新，判断是否处理m+1条WOM之前，前m条WOM出现的累计概率
        np.random.shuffle(self.G.nodes[i]["predecessor"])  # 打乱邻居信息的顺序
        count = 0
        for j in self.G.nodes[i]["predecessor"]:
            if self.G.nodes[i]["WOM_treate_table"][j] != 1:  # 已经处理过的PWOM不再重复进行处理
                p_seq_innovGood, p_seq_innovBad, count = self.process_WOM(
                    i, j, self.G.nodes[j]["state"], p_seq_innovGood, p_seq_innovBad, count
                )

            if count >= self.G.nodes[i]["cognitive_limit"]:
                break

    def set_agent_state(self, i):
        """
        Goal: simulate the decision process of agent i not decided yet in a period.
        Input: i---the name of the agent.
        Output: the state of agent i after decision.
        """
        self.choose_WOM(i)
        # a value generated from the uniform distribution.
        pos_dose = [
            (1 - self.G.nodes[j]["q"])
            for j in self.G.nodes[i]["predecessor"]
            if self.G.nodes[j]["state"] == 1 and self.G.nodes[i]["WOM_treate_table"][j] == 1
        ]
        neg_dose = [
            (1 - self.omega * self.G.nodes[j]["q"])
            for j in self.G.nodes[i]["predecessor"]
            if self.G.nodes[j]["state"] in (-1, -2)
            and self.G.nodes[i]["WOM_treate_table"][j] == 1
            and self.G.nodes[j]["neg_sep"] <= 2
        ]

        if len(pos_dose) != 0:  # if i have predecessors
            pi_pos = 1 - (1 - self.G.nodes[i]["delta"]) * np.cumprod(pos_dose)[-1]
        else:
            pi_pos = self.G.nodes[i]["delta"]

        if len(neg_dose) != 0:
            pi_neg = 1 - np.cumprod(neg_dose)[-1]
        else:
            pi_neg = 0

        alpha = pi_pos / (pi_pos + pi_neg) if pi_pos + pi_neg != 0 else 0
        p_adopt = (1 - pi_neg) * pi_pos + alpha * pi_pos * pi_neg
        p_reject = (1 - pi_pos) * pi_neg + (1 - alpha) * pi_pos * pi_neg
        # p_undecided = (1 - pi_pos)*(1 - pi_neg)

        # TODO 此处更新规则有问题。失望人数很少。
        rand_value = np.random.rand()
        if self.G.nodes[i]["state"] == 0:
            if rand_value < p_adopt:  # 决定采纳
                if self.G.nodes[i]["isDisappointed"]:  # 采纳之后不满意
                    self.G.nodes[i]["next_state"] = -1
                    self.G.nodes[i]["neg_sep"] = 1  # 负面口碑传播的初始距离
                    return -1
                else:  # 采纳之后满意
                    self.G.nodes[i]["next_state"] = 1
                    return 1
            elif p_adopt <= rand_value < p_adopt + p_reject:  # 放弃决定
                self.G.nodes[i]["next_state"] = -2
                # 邻近的最大n_wom_seg
                self.G.nodes[i]["neg_sep"] = (
                    max(
                        [
                            self.G.nodes[j]["neg_sep"]
                            for j in self.G.nodes[i]["predecessor"]
                            if self.G.nodes[j]["state"] in (-1, -2)
                        ]
                    )
                    + 1
                )
                return -2
            else:  # 不改变状态
                return 0

    def get_next_states(self):
        """
        Goal: simulate the decision of all agents not decided yet.
            It is for recording the non-acccumulative number of each states in each period.
        Output: The states of next period.
        """
        next_states = []
        for i in self.G.nodes():
            if self.G.nodes[i]["state"] == 0:
                next_states.append(self.set_agent_state(i))
        return next_states

    def update_states(self):
        """
        Goal: update the state for next period.
        """
        for i in self.G.nodes():
            if self.G.nodes[i]["state"] == 0:
                self.G.nodes[i]["state"] = self.G.nodes[i]["next_state"]

    def diffuse(self, max_steps=50):
        """
        Goal: a single diffusion process.
        Output: the lists of revenue, #adopters, #disappointers, #rejecters across the period.
        """
        num_adopter_list = []
        num_disappointer_list = []
        num_rejecter_list = []
        self.initilize_state()
        for step in range(1, max_steps + 1):
            num_adopter, num_disappointer, num_rejecter = 0, 0, 0
            next_states = self.get_next_states()
            for s in next_states:
                if s == 1:
                    num_adopter += 1
                elif s == -1:
                    num_disappointer += 1
                elif s == -2:
                    num_rejecter += 1
                else:
                    pass
            num_adopter_list.append(num_adopter)
            num_disappointer_list.append(num_disappointer)
            num_rejecter_list.append(num_rejecter)

            self.update_states()
            if sum([1 for i in self.G.nodes if self.G.nodes[i]["state"] == 1]) / self.G.number_of_nodes() > 0.95:
                print(f"策略:{self.seed_type}, 进行至{step}个时间步结束!")
                break
        else:
            print(f"策略:{self.seed_type}, 进行至指定时间步({max_steps})结束!")

        return num_adopter_list, num_disappointer_list, num_rejecter_list

    def multi_diffuse(self, max_steps=50, num_simu=10):
        """
        Goal: perform #num_simu simulations.
        Output: the lists for each list of #adopters, #disappointers, #rejecters across the period.
        """
        num_adopter_cont, num_disappointer_cont, num_rejecter_cont = [], [], []
        print("Simulation begin")
        for i in range(num_simu):
            t1 = time.perf_counter()
            print(f"第{i+1}次模拟", end="  ")
            num_adopter_list, num_disappointer_list, num_rejecter_list = self.diffuse(max_steps=max_steps)
            num_adopter_cont.append(num_adopter_list)
            num_disappointer_cont.append(num_disappointer_list)
            num_rejecter_cont.append(num_rejecter_list)
            print(f"    耗时{time.perf_counter()-t1 : .2f}s")

        print("End")
        return {
            "num_adopter": num_adopter_cont,
            "num_disappointer": num_disappointer_cont,
            "num_rejecter": num_rejecter_cont,
        }

    def multi_diffuse_cores(self, max_steps=50, num_simu=10):
        num_adopter_cont, num_disappointer_cont, num_rejecter_cont = [], [], []
        if NUM_CORNS >= 2:
            pool = multiprocessing.Pool(processes=NUM_CORNS - 1)
            result = []
            for _ in range(num_simu):
                a = pool.apply_async(self.diffuse, kwds={"max_steps": max_steps})
                result.append(a)

            pool.close()
            pool.join()
            for res in result:
                data = res.get()
                num_adopter_cont.append(data[0])
                num_disappointer_cont.append(data[1])
                num_rejecter_cont.append(data[2])
        else:
            for _ in range(num_simu):
                data = self.diffuse(max_steps)
                num_adopter_cont.append(data[0])
                num_disappointer_cont.append(data[1])
                num_rejecter_cont.append(data[2])

        return {
            "num_adopter": num_adopter_cont,
            "num_disappointer": num_disappointer_cont,
            "num_rejecter": num_rejecter_cont,
        }


def convert_result(res_dict, num_simu, max_num_steps):
    """
    Goal: transfer list to np.array
    """
    num_adopter = res_dict["num_adopter"]
    num_disappointer = res_dict["num_disappointer"]
    num_rejecter = res_dict["num_rejecter"]

    adopter_array = np.zeros((num_simu, max_num_steps), dtype=float)
    disappointer_array = np.zeros((num_simu, max_num_steps), dtype=float)
    rejecter_array = np.zeros((num_simu, max_num_steps), dtype=float)

    # 转换为np.array对象
    for i in range(num_simu):
        for j in range(max_num_steps):
            try:
                adopter_array[i][j] = num_adopter[i][j]
                disappointer_array[i][j] = num_disappointer[i][j]
                rejecter_array[i][j] = num_rejecter[i][j]
            except IndexError:
                break

    return {"num_adopter": adopter_array, "num_disappointer": disappointer_array, "num_rejecter": rejecter_array}


def main(delta, mean_q, perc_disp_ratio):
    abms = AgentDiffusionModel(delta=delta, mean_q=mean_q, perc_disp_ratio=perc_disp_ratio)
    num_simu = 10
    res_dict = abms.multi_diffuse(num_simu=num_simu)
    res_dict = convert_result(res_dict, num_simu=num_simu, max_num_steps=30)
    print(res_dict)


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    delta = 0.005
    mean_q = 0.1
    perc_disp_ratio = 0.01
    u_adopt = 1
    u_reject = 1
    lower_d = -3
    upper_d = 5
    seed_type = "social hub"
    G = nx.barabasi_albert_graph(1000, 3)
    abms = AgentDiffusionModel(
        delta=delta,
        mean_q=mean_q,
        perc_disp_ratio=perc_disp_ratio,
        lower_d=lower_d,
        upper_d=upper_d,
        seed_type=seed_type,
        G=G,
        u_adopt=u_adopt,
        u_reject=u_reject,
    )

    max_steps, num_simu = 60, 10
    t1 = time.perf_counter()
    res_dict = abms.multi_diffuse_cores(max_steps=max_steps, num_simu=num_simu)
    res_dict = convert_result(res_dict, max_num_steps=max_steps, num_simu=num_simu)

    mean_num_adopter = np.mean(res_dict["num_adopter"], axis=0)
    mean_num_disappointer = np.mean(res_dict["num_disappointer"], axis=0)
    mean_num_rejecter = np.mean(res_dict["num_rejecter"], axis=0)
    print(f"总共耗时: {time.perf_counter() - t1: .2f}s")
    print(
        f"总采纳人数: {np.sum(mean_num_adopter):.1f}, 总失望人数: {np.sum(mean_num_disappointer):.1f}, 总拒绝人数: {np.sum(mean_num_rejecter): .1f}"
    )

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Number of agents")
    ax.plot(np.arange(max_steps), mean_num_adopter, "go-", lw=1.5, label="Adopter")
    ax.plot(np.arange(max_steps), mean_num_disappointer, "r-", lw=0.5, alpha=0.5, label="Disappointer")
    ax.plot(np.arange(max_steps), mean_num_rejecter, "b-", lw=0.5, alpha=0.5, label="rejecter")
    ax.legend(loc="best")
    plt.show()

