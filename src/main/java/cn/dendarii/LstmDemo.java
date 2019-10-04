package cn.dendarii;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;
import java.util.stream.Collectors;

@SuppressWarnings("unused")
public class LstmDemo {
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
        // 构造lstm的单元，注意这里严格来说内部向量的权重并不要求一定一样，但是出于简化考虑都进行统一
        int dim = 3;
        class lstm {
            // w矩阵和b矩阵
            public double[][] wfh = new double[dim][dim];
            public double[][] wfx = new double[dim][word_num];
            public double[] bf = new double[dim];
            public double[][] wih = new double[dim][dim];
            public double[][] wix = new double[dim][word_num];
            public double[] bi = new double[dim];
            public double[][] wch = new double[dim][dim];
            public double[][] wcx = new double[dim][word_num];
            public double[] bc = new double[dim];
            public double[][] woh = new double[dim][dim];
            public double[][] wox = new double[dim][word_num];
            public double[] bo = new double[dim];
            public double[][] wy = new double[dim][word_num];
            public double[] by = new double[word_num];
            // 状态矩阵
            public double[] f = new double[dim];
            public double[] i = new double[dim];
            public double[] c_prev = new double[dim];
            public double[] c = new double[dim];
            public double[] c_ = new double[dim];
            public double[] h_prev = new double[dim];
            public double[] h = new double[dim];
            public double[] o = new double[dim];
            public double[] y = new double[word_num];
        }
        // 构造lstm单元序列，认为序列长度为3
        int layer_num = 1;
        lstm[] layer = new lstm[layer_num];
        Random rand2 = new Random(2);
        for (int l = 0; l < layer_num; l++) {
            layer[l] = new lstm();
            Mat.apply(layer[l].wfh, x -> rand2.nextDouble(), false);
            Mat.apply(layer[l].wfx, x -> rand2.nextDouble(), false);
            Mat.apply(layer[l].bf, x -> rand2.nextDouble(), false);
            Mat.apply(layer[l].wih, x -> rand2.nextDouble(), false);
            Mat.apply(layer[l].wix, x -> rand2.nextDouble(), false);
            Mat.apply(layer[l].bi, x -> rand2.nextDouble(), false);
            Mat.apply(layer[l].wch, x -> rand2.nextDouble(), false);
            Mat.apply(layer[l].wcx, x -> rand2.nextDouble(), false);
            Mat.apply(layer[l].bc, x -> rand2.nextDouble(), false);
            Mat.apply(layer[l].woh, x -> rand2.nextDouble(), false);
            Mat.apply(layer[l].wox, x -> rand2.nextDouble(), false);
            Mat.apply(layer[l].bo, x -> rand2.nextDouble(), false);
            Mat.apply(layer[l].wy, x -> rand2.nextDouble(), false);
            Mat.apply(layer[l].by, x -> rand2.nextDouble(), false);
            Mat.apply(layer[l].c, x -> rand2.nextDouble(), false);
            Mat.apply(layer[l].h, x -> rand2.nextDouble(), false);
            Mat.normalize2d(layer[l].c);
            Mat.normalize2d(layer[l].h);
        }
        class LayerFunctions {
            public void forward(lstm l,
                                double[] x_t_in,
                                boolean output) {
                if (epouch == 500) {
                    // System.out.println(123);
                }
                // 计算f
                l.f = Mat.add(Mat.mul(l.wfh, l.h_prev), Mat.mul(l.wfx, x_t_in), l.bf);
                Mat.apply(l.f, x -> Mat.sigmoid(x), false);
                // 计算i
                l.i = Mat.add(Mat.mul(l.wih, l.h_prev), Mat.mul(l.wix, x_t_in), l.bi);
                Mat.apply(l.i, x -> Mat.sigmoid(x), false);
                // 计算c_
                l.c_ = Mat.add(Mat.mul(l.wch, l.h_prev), Mat.mul(l.wcx, x_t_in), l.bc);
                Mat.apply(l.c_, x -> Math.tanh(x), false);
                // 计算c
                System.arraycopy(l.c, 0, l.c_prev, 0, l.c.length);
                l.c = Mat.add(Mat.applyMul(l.f, l.c_prev), Mat.applyMul(l.i, l.c_));
                // 计算o
                l.o = Mat.add(Mat.mul(l.woh, l.h_prev), Mat.mul(l.wox, x_t_in), l.bo);
                Mat.apply(l.o, x -> Mat.sigmoid(x), false);
                // 计算h
                System.arraycopy(l.h, 0, l.h_prev, 0, l.h.length);
                l.h = Mat.apply(l.c, x -> Math.tanh(x), true);
                l.h = Mat.applyMul(l.o, l.h);
                // 计算y
                l.y = Mat.add(Mat.mul(Mat.reverse(l.wy), l.h), l.by);
                Mat.softmax(l.y);
            }
            public double[] backward(lstm l,
                                     double[] x_t_in,
                                     double[] dy,
                                     boolean output) {
                if (epouch == 50000) {
                    // System.out.println(123);
                }
                // 1.计算dwy
                double[][] dwy = Mat.mulx1d1d(l.h, dy);
                // 2.计算dh
                double[] dh = Mat.mul(l.wy, dy);
                // 计算do，因关键字原因带下划线
                double[] do_ = Mat.applyMul(l.c, dh);
                do_ = Mat.applyMul(do_, Mat.sigmoidg(l.o));
                // 计算dwoh
                double[][] dwoh = Mat.mulx1d1d(l.h_prev, do_);
                // 计算dwox
                double[][] dwox = Mat.mulx1d1d(x_t_in, do_);
                // 3.计算dc
                double[] dc = Mat.applyMul(dh, l.o);
                dc = Mat.applyMul(dc, Mat.tanhg(l.c));
                // 4.计算dc_
                double[] dc_ = Mat.applyMul(dc, l.i);
                dc_ = Mat.applyMul(dc_, Mat.tanhg(l.c_));
                // 计算dwch
                double[][] dwch = Mat.mulx1d1d(l.h_prev, dc_);
                // 计算dwcx
                double[][] dwcx = Mat.mulx1d1d(x_t_in, dc_);
                // 5.计算df
                double[] df = Mat.applyMul(dc, l.c_prev);
                df = Mat.applyMul(df, Mat.sigmoidg(l.f));
                // 计算dwfh
                double[][] dwfh = Mat.mulx1d1d(l.h_prev, df);
                // 计算dwfx
                double[][] dwfx = Mat.mulx1d1d(x_t_in, df);
                // 6.计算di
                double[] di = Mat.applyMul(dc, l.c_);
                df = Mat.applyMul(df, Mat.sigmoidg(l.i));
                // 计算dwfh
                double[][] dwih = Mat.mulx1d1d(l.h_prev, di);
                // 计算dwfx
                double[][] dwix = Mat.mulx1d1d(x_t_in, di);
                // 更新各个参数
                Mat.update(l.wy, dwy);
                Mat.update(l.by, dy);
                Mat.update(l.wox, Mat.reverse(dwox));
                Mat.update(l.bo, do_);
                Mat.update(l.wch, Mat.reverse(dwch));
                Mat.update(l.wcx, Mat.reverse(dwcx));
                Mat.update(l.bc, dc_);
                Mat.update(l.wfh, Mat.reverse(dwfh));
                Mat.update(l.wfx, Mat.reverse(dwfx));
                Mat.update(l.bf, df);
                Mat.update(l.wih, Mat.reverse(dwih));
                Mat.update(l.wix, Mat.reverse(dwix));
                Mat.update(l.bi, di);
                // 计算上一层的dx，取f、i、c_、o的前向x误差平均值
                double[] dx = new double[word_num];
                for (int d_rol = 0; d_rol < word_num; d_rol++) {
                    for (int d_col = 0; d_col < dim; d_col++) {
                        dx[d_rol] += l.wfx[d_col][d_rol] * df[d_col];
                        dx[d_rol] += l.wix[d_col][d_rol] * di[d_col];
                        dx[d_rol] += l.wcx[d_col][d_rol] * dc_[d_col];
                        dx[d_rol] += l.wox[d_col][d_rol] * do_[d_col];
                    }
                    dx[d_rol] = dx[d_rol] / 4;
                }
                // // 尝试归一化h和c
                // Mat.normalize2d(l.c);
                // Mat.normalize2d(l.h);
                return dx;
            }
        }
        LayerFunctions layerFunctions = new LayerFunctions();
        for (epouch = 0; epouch < 1000000; epouch++) {
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
                loss -= Math.log(layer[layer_num - 1].y[x_t_1_pos]);
                if (layer_num > 1) {
                    // 计算最后一层的权重值，dy=o[t]-x[t+1]
                    double[] dy = Mat.applyMinus(layer[layer_num - 1].y, one_hot[word_map.get(train_list.get(i + 1))]);
                    double[] dx = layerFunctions.backward(layer[layer_num - 1], layer[layer_num - 2].o, dy, false);
                    // 依次更新前几层权重值
                    for (int l_id = layer_num - 2; l_id > 0; l_id--) {
                        dx = layerFunctions.backward(layer[l_id], layer[l_id - 1].o, dx, false);
                    }
                    // 更新第一层
                    layerFunctions.backward(layer[0], x_t_in, dx, true);
                } else {
                    double[] dy = Mat.applyMinus(layer[0].y, one_hot[word_map.get(train_list.get(i + 1))]);
                    layerFunctions.backward(layer[0], x_t_in, dy, epouch % 100 == 0 && i == train_list.size() - 2);
                }
            }
            // 输出损失
            if (false && epouch % 100 == 0) {
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
                double[][][] h_cache = new double[4][layer_num][dim];
                for (int i = 0; i < layer_num; i++) {
                    h_cache[0][i] = layer[i].h;
                    h_cache[1][i] = layer[i].h_prev;
                    h_cache[2][i] = layer[i].c;
                    h_cache[3][i] = layer[i].c_prev;
                }
                StringBuffer sb2 = new StringBuffer();
                for (int i = 0; i < test_list.size(); i++) {
                    // 获取w的one-hot值
                    int x_t_pos = word_map.get(test_list.get(i));
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
                    sb2.append(x_t_pos).append("\t");
                    for (int w = 0; w < word_num; w++) {
                        sb2.append((int) (layer[layer_num - 1].y[w] * 100)).append("\t");
                        if (layer[layer_num - 1].y[w] > layer[layer_num - 1].y[max]) {
                            max = w;
                        }
                    }
                    sb2.append("\n");
                    sb.append(word_seq[max]);
                    maxVal += layer[layer_num - 1].y[max];
                }
                // 恢复layer的临时状态
                for (int i = 0; i < layer_num; i++) {
                    layer[i].h = h_cache[0][i];
                    layer[i].h_prev = h_cache[1][i];
                    layer[i].c = h_cache[2][i];
                    layer[i].c_prev = h_cache[3][i];
                }
                sb.append("\tavgscore=").append(df.format(maxVal));
                System.out.println(sb.toString());
                if (false) {
                    System.out.println(sb2.toString());
                }
            }
            if (true && epouch % 10000 == 0) {
                System.out.print(epouch + "\t" + df.format(loss) + "\t");
                System.out.print(Mat.vectorString("h", df, layer[0].h));
                System.out.print("\t" + Mat.vectorString("c", df, layer[0].c) + "\n");
            }
        }
    }
}
