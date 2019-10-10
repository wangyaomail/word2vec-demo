package cn.dendarii;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;
import java.util.stream.Collectors;

import cn.dendarii.data.CountList;
import cn.dendarii.data.Pair;
import cn.dendarii.model.Lstm;

/**
 * seq2seq整体和单层lstm差不多，只不过每层的输入和输出使用分开的两个lstm单元，中间靠c值传递
 */
@SuppressWarnings("unused")
public class Seq2SeqDemo {
    static String[] nauty = { "x", "y", "z" };// , "a", "b", "c", "d", "e", "f" };
    static String[] pattern = { "ab-cd", "ac-bd", "bc-ad", "cd-ab" };
    static String startToken = "s";
    static DecimalFormat df = new DecimalFormat("#0.00000");
    static int epouch;
    public static void main(String[] args) {
        HashMap<String, Integer> word_count = new HashMap<String, Integer>();
        // 构造数据集，构造为：混淆+pattern+混淆=pattern的方式
        // 这里扩大了nauty的集合，意味着混淆会更明显
        List<Pair<List<String>, List<String>>> train_list = new ArrayList<Pair<List<String>, List<String>>>();
        Random rand1 = new Random(1);
        class DataMaker {
            public void makeDataSet(List<Pair<List<String>, List<String>>> dataset,
                                    int rownum) {
                for (int turn = 0; turn < rownum; turn++) {
                    CountList<String> input = new CountList<String>(word_count);
                    CountList<String> output = new CountList<String>(word_count);
                    String[] patternSrc = pattern[turn % pattern.length].split("-");
                    input.add(nauty[rand1.nextInt(nauty.length)]);
                    for (String p1 : patternSrc[0].split("")) {
                        input.add(p1);
                    }
                    input.add(nauty[rand1.nextInt(nauty.length)]);
                    output.add(startToken);
                    for (String p2 : patternSrc[1].split("")) {
                        output.add(p2);
                    }
                    dataset.add(new Pair<List<String>, List<String>>(input, output));
                }
            }
        }
        DataMaker dataMaker = new DataMaker();
        // 生成训练数据，这里训练数据后面应该变为pair的形式，一对一对出现，中间用字符s来隔开
        dataMaker.makeDataSet(train_list, pattern.length * 4);
        // 输出数据集
        if (false) {
            train_list.forEach(w -> System.out.println(w));
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
        // 按照字典序排序计算每个词的one-hot值
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
        // 构造测试集，测试集和
        List<Pair<List<String>, List<String>>> test_list = new ArrayList<Pair<List<String>, List<String>>>();
        dataMaker.makeDataSet(test_list, pattern.length);
        if (true) {
            test_list.forEach(w -> System.out.println(w));
        }
        int layer_num = 1;
        Lstm[] encoder = new Lstm[layer_num];
        Lstm[] decoder = new Lstm[layer_num];
        int dim = 20;
        Random rand2 = new Random(2);
        for (int l = 0; l < layer_num; l++) {
            encoder[l] = new Lstm(rand2, dim, word_num);
            decoder[l] = new Lstm(rand2, dim, word_num);
        }
        // 需要按照k-v的形式构造训练过程，和简单lstm的训练区别在于：
        // 1、encoder需要对输入序列整个计算完后，再向decoder传输y、c和h
        // 2、encoder需要保存每一步的c和h
        // 另外需要注意的是：
        // 1、在简单lstm中c和h能通过循环训练来回避初始值的影响，但是在k-v式的训练中c和h的初始值一定会有干扰，具体干扰是什么样的还需要观察
        for (epouch = 0; epouch < 10000000; epouch++) {
            if (epouch == 1000) {
                System.out.println();
            }
            double encoder_dh_length = 0; // 计算encoder的向量长度
            for (int t = 0; t < train_list.size(); t++) {
                Pair<List<String>, List<String>> trainPair = train_list.get(t);
                // 1. 计算encoder的forward
                List<String> xList = trainPair.getK();
                // 训练的第几个字：模型第几层：一共要缓存多少个状态：维度
                double[][][][] encoder_state = new double[xList.size()][layer_num][2][dim];
                double[][][][] encoder_io = new double[xList.size()][layer_num][2][word_num];
                for (int x = 0; x < xList.size(); x++) {
                    // 获取x的one-hot值
                    int x_t_pos = word_map.get(xList.get(x));
                    double[] x_t_in = one_hot[x_t_pos];
                    // 把x送入encoder，注意序列开头的h和c矩阵
                    if (x == 0) {
                        for (int l = 0; l < layer_num; l++) {
                            Mat.apply(encoder_state[x][l][0], w -> rand2.nextDouble(), false);
                            Mat.normalize2d(encoder_state[x][l][0]);
                            Mat.apply(encoder_state[x][l][1], w -> rand2.nextDouble(), false);
                            Mat.normalize2d(encoder_state[x][l][1]);
                        }
                    }
                    Mat.arraycopy(x_t_in, encoder_io[x][0][0], word_num);
                    encoder[0].forward(x_t_in, encoder_state[x][0][0], encoder_state[x][0][1], false);
                    Mat.arraycopy(encoder[0].y, encoder_io[x][0][0], word_num);
                    if (layer_num > 1) {
                        // 依次计算后面几层
                        for (int l = 1; l < layer_num; l++) {
                            Mat.arraycopy(encoder[l - 1].h, encoder_state[x][l][0], dim);
                            Mat.arraycopy(encoder[l - 1].c, encoder_state[x][l][1], dim);
                            Mat.arraycopy(encoder[l - 1].y, encoder_io[x][l][0], word_num);
                            encoder[l].forward(encoder[l - 1].y, encoder_state[x][l][0], encoder_state[x][l][1], true);
                            Mat.arraycopy(encoder[l].y, encoder_io[x][l][1], word_num);
                        }
                    }
                }
                // 2. 计算decoder的forward
                // 训练的第几个字：模型第几层：一共要缓存多少个状态：维度（维度取最大值，0-hprev，1-cprev，2-input，3-output）
                List<String> yList = trainPair.getV();
                double[][][][] decoder_state = new double[yList.size()][layer_num][2][dim];
                double[][][][] decoder_io = new double[yList.size()][layer_num][2][word_num];
                // 注意序列的最后一个字不参与训练
                for (int y = 0; y < yList.size() - 1; y++) {
                    // 获取y的one-hot值
                    int y_t_pos = word_map.get(yList.get(y));
                    double[] y_t_in = one_hot[y_t_pos];
                    // 把y送入decoder
                    for (int l = 0; l < layer_num; l++) {
                        if (y == 0) {
                            // 将encoder最后一层的值送入decoder的每一层
                            Mat.arraycopy(encoder[layer_num - 1].h, decoder_state[0][l][0], dim);
                            Mat.arraycopy(encoder[layer_num - 1].c, decoder_state[0][l][1], dim);
                        } else {
                            // 将该层上一层的状态送入这一层
                            Mat.arraycopy(decoder[l].h, decoder_state[y][l][0], dim);
                            Mat.arraycopy(decoder[l].c, decoder_state[y][l][1], dim);
                        }
                    }
                    Mat.arraycopy(y_t_in, decoder_io[y][0][0], word_num);
                    decoder[0].forward(y_t_in, decoder_state[y][0][0], decoder_state[y][0][1], false);
                    Mat.arraycopy(decoder[0].y, decoder_io[y][0][1], word_num);
                    if (layer_num > 1) {
                        // 依次计算后面几层
                        for (int l = 1; l < layer_num; l++) {
                            Mat.arraycopy(decoder[l - 1].y, decoder_io[y][l][0], word_num); // 这样是重复存储，但逻辑上更工整
                            decoder[l].forward(decoder[l - 1].y, decoder_state[y][l][0], decoder_state[y][l][1], true);
                            Mat.arraycopy(decoder[y].y, decoder_io[y][l][1], word_num);
                        }
                    }
                }
                // 3.计算decoder的backward
                Lstm dlstm = new Lstm(dim, word_num);
                double[] encoder_dh = new double[dim];
                for (int y = yList.size() - 2; y >= 0; y--) {
                    double[] dy = Mat.applyMinus(decoder_io[y][layer_num - 1][1],
                                                 one_hot[word_map.get(yList.get(y + 1))]);
                    if (layer_num > 1) {
                        for (int l = layer_num - 1; l >= 0; l--) {
                            if (y == 0 && l == 0) {
                                // 如果eos的位置，且是第一个输出层，那么要返回encoder给dh
                                encoder_dh = decoder[l].backward(decoder_io[y][l][0],
                                                                 dy,
                                                                 decoder_state[y][l][0],
                                                                 decoder_state[y][l][1],
                                                                 dlstm,
                                                                 false,
                                                                 false);
                            } else {
                                dy = decoder[l].backward(decoder_io[y][l][0],
                                                         dy,
                                                         decoder_state[y][l][0],
                                                         decoder_state[y][l][1],
                                                         dlstm,
                                                         true,
                                                         false);
                            }
                        }
                    } else {
                        encoder_dh = decoder[0].backward(decoder_io[y][0][0],
                                                         dy,
                                                         decoder_state[y][0][0],
                                                         decoder_state[y][0][1],
                                                         dlstm,
                                                         y != 0,
                                                         false);
                    }
                }
                // 4.更新decoder的参数
                for (int l = 0; l < layer_num; l++) {
                    decoder[l].update(dlstm);
                }
                if (false) { // encoder传回去的梯度关闭
                    // 5.计算encoder的backward
                    // 这里采用通过h传递梯度的策略
                    dlstm = new Lstm(dim, word_num);
                    for (int x = xList.size() - 1; x >= 0; x--) {
                        for (int l = layer_num - 1; l >= 0; l--) {
                            encoder_dh_length += Mat.vectorLength(encoder_dh);
                            encoder_dh = encoder[l].backwardByDh(encoder_io[x][l][0],
                                                                 encoder_dh,
                                                                 encoder_state[x][l][0],
                                                                 encoder_state[x][l][1],
                                                                 dlstm,
                                                                 false,
                                                                 false);
                        }
                    }
                    // 6.更新encoder的参数
                    for (int l = 0; l < layer_num; l++) {
                        encoder[l].update(dlstm);
                    }
                }
            }
            if (epouch % 100 == 0) {
                System.out.println("dh=" + df.format(encoder_dh_length));
            }
            // 利用测试集计算损失
            for (int t = 0; t < test_list.size(); t++) {
                Pair<List<String>, List<String>> testPair = test_list.get(t);
                double[] h_prev = new double[dim];
                double[] c_prev = new double[dim];
                List<String> xList = testPair.getK();
                for (int x = 0; x < xList.size(); x++) {
                    double[] x_t_in = one_hot[word_map.get(xList.get(x))];
                    encoder[0].forward(x_t_in, h_prev, c_prev, false);
                    h_prev = encoder[0].h;
                    c_prev = encoder[0].c;
                    if (layer_num > 1) {
                        for (int l = 1; l < layer_num; l++) {
                            encoder[l].forward(encoder[l - 1].y, h_prev, c_prev, true);
                            h_prev = encoder[l].h;
                            c_prev = encoder[l].c;
                        }
                    }
                }
                List<String> yList = testPair.getV();
                double loss = 0;
                double maxVal = 0.0;
                List<String> yPredict = new ArrayList<String>();
                for (int y = 0; y < yList.size() - 1; y++) {
                    double[] y_t_in = one_hot[word_map.get(yList.get(y))];
                    decoder[0].forward(y_t_in, h_prev, c_prev, false);
                    h_prev = decoder[0].h;
                    c_prev = decoder[0].c;
                    if (layer_num > 1) {
                        for (int l = 1; l < layer_num; l++) {
                            decoder[l].forward(decoder[l - 1].y, h_prev, c_prev, true);
                            h_prev = encoder[l].h;
                            c_prev = encoder[l].c;
                        }
                    }
                    int max = 0;
                    for (int w = 0; w < word_num; w++) {
                        if (decoder[layer_num - 1].y[w] > decoder[layer_num - 1].y[max]) {
                            max = w;
                        }
                    }
                    maxVal += decoder[layer_num - 1].y[max];
                    yPredict.add(word_seq[max]);
                    loss -= Math.log(decoder[layer_num - 1].y[word_map.get(yList.get(y + 1))]);
                }
                // 输出损失
                if (true && epouch % 100 == 0) {
                    StringBuffer sb = new StringBuffer();
                    sb.append("epouch=").append(epouch).append("\t");
                    sb.append("loss=").append(df.format(loss)).append("\ttest=");
                    for (int i = 0; i < xList.size(); i++) {
                        sb.append(xList.get(i));
                    }
                    sb.append("\tpredict=");
                    for (int i = 0; i < yPredict.size(); i++) {
                        sb.append(yPredict.get(i));
                    }
                    sb.append("\tavgscore=").append(df.format(maxVal));
                    System.out.println(sb.toString());
                }
            }
        }
    }
}
