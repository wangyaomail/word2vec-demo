package cn.dendarii;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;
import java.util.stream.Collectors;

import cn.dendarii.model.BasicRnn;

/**
 * 测试封装的BasicRnn好不好用
 */
@SuppressWarnings("unused")
public class RnnDemo2 {
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
            word_count_sorted.forEach(e -> System.out.print(e.getKey() + ":" + e.getValue() + ", "));
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
        // 构造测试集
        List<String> test_list = new ArrayList<String>();
        for (int i = 0; i < pattern.length; i++) {
            for (int j = 0; j < pattern[i].length; j++) {
                test_list.add(pattern[i][j]);
            }
        }
        // 首先构造RNN基本单元
        int dim = 5;
        // 构造rnn单元序列，认为序列长度为3
        int layer_num = 2;
        BasicRnn[] layer = new BasicRnn[layer_num];
        Random rand2 = new Random(2);
        for (int l = 0; l < layer_num; l++) {
            layer[l] = new BasicRnn(rand2, dim, word_num, true);
        }
        // 进行训练
        // 更新每个epouch中训练的数据都是环形的，且所有w权重都从0开始
        for (epouch = 0; epouch < 1000000; epouch++) {
            double loss = 0;
            double[][] h_prev_layers = new double[layer_num][dim];
            // 注意学习的时候，序列最后一位不要了，保证循环性
            for (int i = 0; i < train_list.size() - 1; i++) {
                // 获取w的one-hot值
                double[] x_t_in = one_hot[word_map.get(train_list.get(i))];
                for (int l = 0; l < layer_num; l++) {
                    h_prev_layers[l] = layer[l].h;
                    layer[l].forward(l > 0 ? layer[l - 1].y : x_t_in, h_prev_layers[l], true);
                }
                // 计算损失，这一步的损失应该是当前词送到模型以后的输出和下一位之间的交叉熵
                int x_t_1_pos = word_map.get(train_list.get(i + 1));
                loss -= Math.log(layer[layer_num - 1].y[x_t_1_pos]);
                // 反向传播
                double[] y_expected = one_hot[word_map.get(train_list.get(i + 1))];
                double[] dy = Mat.applyMinus(layer[layer_num - 1].y, y_expected);
                for (int l = layer_num - 1; l >= 0; l--) {
                    BasicRnn drnn = new BasicRnn(dim, word_num);
                    dy = layer[l].backward(l > 0 ? layer[l - 1].y : x_t_in, dy, h_prev_layers[l], drnn, true);
                    layer[l].update(drnn);
                }
            }
            // 输出损失
            if (epouch % 100 == 0) {
                StringBuffer sb = new StringBuffer();
                sb.append("epouch=").append(epouch).append("\t");
                sb.append("loss=").append(df.format(loss)).append("\ttest=");
                for (int i = 0; i < test_list.size(); i++) {
                    sb.append(test_list.get(i));
                }
                sb.append("\tpredict=");
                // 进行预测
                double maxVal = 0.0;
                // 保存layer的临时状态
                double[][] h_cache = new double[layer_num][dim];
                for (int i = 0; i < layer_num; i++) {
                    h_cache[i] = layer[i].h;
                }
                for (int i = 0; i < test_list.size(); i++) {
                    // 获取w的one-hot值
                    double[] x_t_in = one_hot[word_map.get(test_list.get(i))];
                    // 先计算第一层
                    h_prev_layers[0] = layer[0].h;
                    layer[0].forward(x_t_in, h_prev_layers[0], true);
                    if (layer_num > 1) {
                        // 依次计算后面几层
                        for (int l = 1; l < layer_num; l++) {
                            h_prev_layers[l] = layer[l].h;
                            layer[l].forward(layer[l - 1].y, h_prev_layers[l], true);
                        }
                    }
                    int max = 0;
                    for (int w = 0; w < word_num; w++) {
                        if (layer[layer_num - 1].y[w] > layer[layer_num - 1].y[max]) {
                            max = w;
                        }
                    }
                    sb.append(word_seq[max]);
                    maxVal += layer[layer_num - 1].y[max];
                }
                // 恢复layer的临时状态
                for (int i = 0; i < layer_num; i++) {
                    layer[i].h = h_cache[i];
                }
                sb.append("\tavgscore=").append(df.format(maxVal / test_list.size()));
                // 输出第一层的h
                sb.append("\th=");
                for (int i = 0; i < h_cache[0].length; i++) {
                    sb.append(df.format(h_cache[0][i])).append(",");
                }
                System.out.println(sb.toString());
            }
        }
    }
}
