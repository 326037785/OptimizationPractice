function pso_1D()
    clc;
    clear;
    close all;

    % 参数设置
    num_particles = 30; % 粒子数量
    num_iterations = 100; % 迭代次数
    c1 = 1.5; % 个体学习因子
    c2 = 1.5; % 社会学习因子
    w = 0.5; % 惯性权重

    % 粒子初始化
    x = rand(num_particles, 1) * 10 - 5; % 初始位置，范围[-5, 5]
    v = rand(num_particles, 1) * 2 - 1; % 初始速度，范围[-1, 1]

    % 粒子适应度初始化
    pbest = x; % 个体最优位置
    pbest_val = obj_function(pbest); % 个体最优适应度

    % 全局最优初始化
    [gbest_val, idx] = min(pbest_val);
    gbest = pbest(idx);

    % PSO主循环
    for iter = 1:num_iterations
        % 更新速度和位置
        r1 = rand(num_particles, 1);
        r2 = rand(num_particles, 1);
        v = w * v + c1 * r1 .* (pbest - x) + c2 * r2 .* (gbest - x);
        x = x + v;

        % 更新个体最优
        new_pbest_val = obj_function(x);
        update_idx = new_pbest_val < pbest_val;
        pbest(update_idx) = x(update_idx);
        pbest_val(update_idx) = new_pbest_val(update_idx);

        % 更新全局最优
        [new_gbest_val, idx] = min(pbest_val);
        if new_gbest_val < gbest_val
            gbest_val = new_gbest_val;
            gbest = pbest(idx);
        end
    end

    % 绘图
    x_values = -5:0.1:5;
    y_values = obj_function(x_values);
    plot(x_values, y_values, 'b-', 'LineWidth', 1.5);
    hold on;
    plot(gbest, gbest_val, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
    legend('Objective Function', 'Global Best');
    title('PSO in 1D');
    xlabel('x');
    ylabel('f(x)');
    grid on;
end

function y = obj_function(x)
    % 加入噪声的目标函数
    noise = 0.1 * randn(size(x));
    y = (x - 2).^2 + noise;
end
