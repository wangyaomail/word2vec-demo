package cn.dendarii;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;
import java.util.stream.Collectors;

/**
 * 简化版的rnn，每两步训练之间的变量用下划线命名，而中间变量用驼峰命名
 */
@SuppressWarnings("unused")
public class RnnDemo {
    static String[] nauty = { "x", "y", "z" };
    static String[][] pattern = { { "a", "b", "c" }, { "d", "e", "f" } };
    static DecimalFormat df = new DecimalFormat("#0.00000");
    static int epouch;
    public static void main(String[] args) {
        // 构造数据集
        List<String> train_list = new ArrayList<String>();
        if (true) {
            // 生成需要的训练字符串，格式是混淆字符串+训练字符串1+混淆字符串+训练字符串2+...
            Random rand1 = new Random(1);
            for (int turn = 0; turn < 20; turn++) {
                int nauty_num = rand1.nextInt(2);
                for (int i = 0; i < nauty_num; i++) {
                    train_list.add(nauty[rand1.nextInt(nauty.length)]);
                }
                String[] insertPattern = pattern[turn % 2]; // 如果是4则是1:1:1:1，如果是5则是3:3:2:2
                for (int i = 0; i < insertPattern.length; i++) {
                    train_list.add(insertPattern[i]);
                }
            }
        } else if (false) {
            // 生成需要的训练字符串，格式是训练字符串1+训练字符串2+训练字符串1+训练字符串2...
            for (int turn = 0; turn < 4; turn++) {
                String[] insertPattern = pattern[turn % 2];
                for (int i = 0; i < insertPattern.length; i++) {
                    train_list.add(insertPattern[i]);
                }
            }
        }
        // 将序列第一个值加入序列末尾，保证循环性
        train_list.add(pattern[0][0]);
        // 输出数据集
        if (true) {
            train_list.forEach(w -> System.out.print(w + " "));
            System.out.println("");
        }
        // 对数据集统计词频
        HashMap<String, Integer> word_count = new HashMap<String, Integer>();
        for (String word : train_list) {
            Integer countInteger = word_count.get(word);
            if (countInteger != null) {
                word_count.put(word, countInteger + 1);
            } else {
                word_count.put(word, 1);
            }
        }
        int word_num = word_count.size();
        // 排序
        List<Entry<String, Integer>> word_count_sorted = word_count.entrySet()
                                                                   .stream()
                                                                   .sorted(Comparator.comparing(e -> -e.getValue()))
                                                                   .collect(Collectors.toList());
        // 输出word_count
        if (true) {
            word_count_sorted.forEach(e -> System.out.print(e.getKey() + ":" + e.getValue()
                    + ", "));
            System.out.println("");
        }
        // 按照词频排序计算每个词的one-hot值
        String[] word_seq = new String[word_num];
        double[][] one_hot = new double[word_num][word_num];
        HashMap<String, Integer> word_map = new HashMap<String, Integer>(word_num);
        for (int i = 0; i < word_num; i++) {
            word_seq[i] = word_count_sorted.get(i).getKey();
            one_hot[i][i] = 1.0;
            word_map.put(word_seq[i], i);
        }
        // 输出one_hot
        if (true) {
            for (int i = 0; i < word_num; i++) {
                System.out.print(i + ":" + word_seq[i] + ", ");
            }
            System.out.println("");
            for (int i = 0; i < word_num; i++) {
                System.out.print(word_seq[i]);
                for (int j = 0; j < word_num; j++) {
                    System.out.print("\t" + one_hot[i][j]);
                }
                System.out.println("");
            }
        }
        // 首先构造RNN基本单元
        int dim = 5;
        class rnn {
            // 三个w矩阵和两个偏置项，大小都是向量维度*向量维度，这里默认中间向量维度和输出向量维度是一样的
            public double[][] wx = new double[dim][word_num];
            public double[][] wh = new double[dim][dim];
            public double[][] wo = new double[word_num][dim];
            // 偏置项，大小是维度*1
            public double[] bh = new double[dim];
            public double[] bo = new double[word_num];
            // h、o状态，大小均是向量维度*1
            public double[] h_prev = new double[dim];
            public double[] h = new double[dim];
            public double[] o = new double[word_num];
            public double[] dh_prev = new double[dim];
            // 持久化更新梯度，注意这些梯度值很奇怪，只会不停变大，不会越变越小
            public double[][] gwx = new double[dim][word_num];
            public double[][] gwh = new double[dim][dim];
            public double[][] gwo = new double[word_num][dim];
            public double[] gbh = new double[dim];
            public double[] gbo = new double[word_num];
        }
        // 构造rnn单元序列，认为序列长度为3
        int layer_num = 1;
        rnn[] layer = new rnn[layer_num];
        Random rand2 = new Random(2);
        for (int l = 0; l < layer_num; l++) {
            layer[l] = new rnn();
            for (int i = 0; i < dim; i++) {
                for (int j = 0; j < word_num; j++) {
                    layer[l].wx[i][j] = rand2.nextDouble();
                    layer[l].wo[j][i] = rand2.nextDouble();
                }
                for (int j = 0; j < dim; j++) {
                    layer[l].wh[i][j] = rand2.nextDouble();
                }
                layer[l].h[i] = rand2.nextDouble();
            }
            Mat.normalize2d(layer[l].h);
        }
        double learnRate = 0.05;
        class LayerFunctions {
            public void forward(rnn l,
                                double[] x_t_in,
                                boolean output) {
                if (epouch == 1000) {
                    System.out.println("123");
                }
                // 计算x
                double[] x_t_out = new double[dim];
                for (int d = 0; d < dim; d++) {
                    for (int j = 0; j < word_num; j++) {
                        x_t_out[d] += l.wx[d][j] * x_t_in[j];
                    }
                }
                // 计算和x加和以前的h
                double[] h_t_in = new double[dim];
                for (int d_row = 0; d_row < dim; d_row++) {
                    for (int d_col = 0; d_col < dim; d_col++) {
                        h_t_in[d_row] += l.wh[d_row][d_col] * l.h[d_col];
                    }
                }
                // 计算最后的h，注意这里并不是h更新的地方
                for (int d = 0; d < dim; d++) {
                    l.h_prev[d] = l.h[d];
                    l.h[d] = Math.tanh(h_t_in[d] + x_t_out[d] + l.bh[d]);
                }
                // // 强行归一化h，让h只能在圆面上移动
                // Mat.normalize2d(l.h);
                if (output) {
                    System.out.print(epouch + "\t");
                    for (int d = 0; d < dim; d++) {
                        System.out.print(df.format(l.h[d]) + ",");
                    }
                    System.out.println("");
                }
                // // 对h进行正则化
                // Mat.normalize2d(l.h);
                // 计算output
                double[] o_t_in = new double[word_num];
                for (int d_row = 0; d_row < word_num; d_row++) {
                    for (int d_col = 0; d_col < dim; d_col++) {
                        o_t_in[d_row] += l.wo[d_row][d_col] * l.h[d_col];
                    }
                }
                for (int d = 0; d < word_num; d++) {
                    l.o[d] = Math.exp(o_t_in[d]);
                }
                // 对l.o进行归一化，等同softmax的被除数
                Mat.normalize1d(l.o);
            }
            public double[] backward(rnn l,
                                     double[] x_t_in,
                                     double[] dy,
                                     boolean output) {
                if (epouch == 1000) {
                    System.out.println("123");
                }
                double[][] dwx = new double[dim][word_num];
                double[][] dwh = new double[dim][dim];
                double[][] dwo = new double[word_num][dim];
                double[] dbh = new double[dim];
                double[] dbo = new double[word_num];
                // 计算dwo
                for (int d_rol = 0; d_rol < word_num; d_rol++) {
                    for (int d_col = 0; d_col < dim; d_col++) {
                        dwo[d_rol][d_col] = dy[d_rol] * l.h[d_col];
                    }
                }
                // 计算dbo
                for (int d = 0; d < word_num; d++) {
                    dbo[d] = dy[d];
                }
                // 计算dh
                double[] dh = new double[dim];
                // 这个地方加不加dh_prev对结果的影响都不大
                // System.arraycopy(l.dh_prev, 0, dh, 0, dim);
                for (int d_rol = 0; d_rol < word_num; d_rol++) {
                    for (int d_col = 0; d_col < dim; d_col++) {
                        dh[d_col] += l.wo[d_rol][d_col] * dy[d_rol];
                    }
                }
                // dh是经过tanh计算的，需要在原有的结果中消除tanh的影响
                double[] dh_raw = new double[dim];
                for (int d = 0; d < dim; d++) {
                    dh_raw[d] = (1 - l.h[d] * l.h[d]) * dh[d];
                }
                // 计算dhprev
                l.dh_prev = new double[dim];
                for (int d_rol = 0; d_rol < dim; d_rol++) {
                    for (int d_col = 0; d_col < dim; d_col++) {
                        l.dh_prev[d_rol] += l.wh[d_rol][d_col] * dh_raw[d_rol];
                    }
                }
                // 计算dwx
                for (int d_rol = 0; d_rol < word_num; d_rol++) {
                    for (int d_col = 0; d_col < dim; d_col++) {
                        dwx[d_col][d_rol] = x_t_in[d_rol] * dh_raw[d_col];
                    }
                }
                // 计算dwh
                for (int d_rol = 0; d_rol < dim; d_rol++) {
                    for (int d_col = 0; d_col < dim; d_col++) {
                        dwh[d_rol][d_col] = l.h_prev[d_rol] * dh_raw[d_col];
                    }
                }
                // 计算dbh
                for (int d = 0; d < dim; d++) {
                    dbh[d] = dh_raw[d];
                }
                // 更新各个值
                Mat.update(l.wo, dwo, learnRate);
                Mat.update(l.wh, dwh, learnRate);
                Mat.update(l.wx, dwx, learnRate);
                Mat.update(l.bo, dbo, learnRate);
                Mat.update(l.bh, dbh, learnRate);
                // 输出各个更新向量的向量长度
                if (output) {
                    System.out.print(df.format(Mat.vectorLength(dwh)) + "\t");
                }
                // // 对当前层所有参数归一化
                // normalize(l);
                // 计算对x的更新要求
                double[] dx = new double[word_num];
                for (int d_rol = 0; d_rol < word_num; d_rol++) {
                    for (int d_col = 0; d_col < dim; d_col++) {
                        dx[d_rol] = l.wx[d_col][d_rol] * dh[d_col];
                    }
                }
                return dx;
            }
            public void normalize(rnn l) {
                Mat.normalize(l.wx, false);
                Mat.normalize(l.wh, false);
                Mat.normalize(l.wo, false);
                Mat.normalize2d(l.bh);
                Mat.normalize2d(l.bo);
                // normalize(l.h);
                // normalize(l.o);
                // normalize(l.gwx);
                // normalize(l.gwh);
                // normalize(l.gwo);
                // normalize(l.gbh);
                // normalize(l.gbo);
            }
        }
        LayerFunctions layerFunctions = new LayerFunctions();
        // 构造测试集
        List<String> testList = new ArrayList<String>();
        for (int i = 0; i < pattern.length; i++) {
            for (int j = 0; j < pattern[i].length; j++) {
                testList.add(pattern[i][j]);
            }
        }
        // 进行训练
        // 更新每个epouch中训练的数据都是环形的，且所有w权重都从0开始
        for (epouch = 0; epouch < 10000; epouch++) {
            double loss = 0;
            // 注意学习的时候，序列最后一位不要了，保证循环性
            for (int i = 0; i < train_list.size() - 1; i++) {
                // 获取w的one-hot值
                int x_t_pos = word_map.get(train_list.get(i));
                double[] x_t_in = one_hot[x_t_pos];
                // 先计算第一层
                layerFunctions.forward(layer[0], x_t_in, false);
                if (layer_num > 1) {
                    // 依次计算后面几层
                    for (int l_id = 1; l_id < layer_num; l_id++) {
                        layerFunctions.forward(layer[l_id], layer[l_id - 1].o, true);
                    }
                }
                // 计算损失，这一步的损失应该是当前词送到模型以后的输出和下一位之间的交叉熵
                int x_t_1_pos = word_map.get(train_list.get(i + 1));
                loss -= Math.log(layer[layer_num - 1].o[x_t_1_pos]);
                if (layer_num > 1) {
                    // 计算最后一层的权重值，dy=o[t]-x[t+1]
                    double[] dy = new double[word_num];
                    double[] o_expected = one_hot[word_map.get(train_list.get(i + 1))];
                    for (int d = 0; d < word_num; d++) {
                        dy[d] = layer[layer_num - 1].o[d] - o_expected[d];
                    }
                    double[] dx = layerFunctions.backward(layer[layer_num - 1],
                                                          layer[layer_num - 2].o,
                                                          dy,
                                                          false);
                    // 依次更新前几层权重值
                    for (int l_id = layer_num - 2; l_id > 0; l_id--) {
                        dx = layerFunctions.backward(layer[l_id], layer[l_id - 1].o, dx, false);
                    }
                    // 更新第一层
                    layerFunctions.backward(layer[0], x_t_in, dx, true);
                } else {
                    double[] dy = new double[word_num];
                    double[] o_expected = one_hot[word_map.get(train_list.get(i + 1))];
                    for (int d = 0; d < word_num; d++) {
                        dy[d] = layer[0].o[d] - o_expected[d];
                    }
                    layerFunctions.backward(layer[0],
                                            x_t_in,
                                            dy,
                                            epouch % 100 == 0 && i == train_list.size() - 2);
                }
            }
            if (false && epouch % 10 == 0) {
                System.out.println("");
            }
            // 输出损失
            if (epouch % 100 == 0) {
                StringBuffer sb = new StringBuffer();
                sb.append("epouch=").append(epouch).append("\t");
                sb.append("loss=").append(df.format(loss)).append("\ttest=");
                for (int i = 0; i < testList.size(); i++) {
                    sb.append(testList.get(i));
                }
                sb.append("\tpredict=");
                // 进行预测
                double maxVal = 0.0;
                // 保存layer的临时状态
                double[][][] h_cache = new double[2][layer_num][dim];
                for (int i = 0; i < layer_num; i++) {
                    h_cache[0][i] = layer[i].h;
                    h_cache[1][i] = layer[i].h_prev;
                }
                for (int i = 0; i < testList.size(); i++) {
                    // 获取w的one-hot值
                    int x_t_pos = word_map.get(testList.get(i));
                    double[] x_t_in = one_hot[x_t_pos];
                    // 先计算第一层
                    layerFunctions.forward(layer[0], x_t_in, false);
                    if (layer_num > 1) {
                        // 依次计算后面几层
                        for (int l_id = 1; l_id < layer_num; l_id++) {
                            layerFunctions.forward(layer[l_id], layer[l_id - 1].o, false);
                        }
                    }
                    int max = 0;
                    for (int w = 0; w < word_num; w++) {
                        if (layer[layer_num - 1].o[w] > layer[layer_num - 1].o[max]) {
                            max = w;
                        }
                    }
                    sb.append(word_seq[max]);
                    maxVal += layer[layer_num - 1].o[max];
                }
                // 恢复layer的临时状态
                for (int i = 0; i < layer_num; i++) {
                    layer[i].h = h_cache[0][i];
                    layer[i].h_prev = h_cache[1][i];
                }
                sb.append("\tavgscore=").append(df.format(maxVal));
                System.out.println(sb.toString());
            }
        }
    }
}
