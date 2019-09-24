package cn.dendarii;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.Map.Entry;
import java.util.stream.Collectors;

@SuppressWarnings("unused")
public class GloveDemo {
    // nauty是混淆用的字符串
    static String[] nauty = { "x", "y", "z" };// , "α", "β","γ", "φ" };
    // pattern是实际需要训练的字符串
    static String[][] pattern = { { "a", "x", "b" },
                                  { "c", "x", "b" },
                                  { "c", "x", "d" },
                                  { "a", "x", "d" },
                                  { "a", "x", "b" } };
    public static void main(String[] args) {
        try {
            List<String> trainList = new ArrayList<String>();
            if (true) {
                // 生成需要的训练字符串，格式是混淆字符串+训练字符串1+混淆字符串+训练字符串2+...
                Random rand1 = new Random(1);
                for (int turn = 0; turn < 20; turn++) {
                    int nauty_num = rand1.nextInt(10);
                    for (int i = 0; i < nauty_num; i++) {
                        trainList.add(nauty[rand1.nextInt(nauty.length)]);
                    }
                    String[] insertPattern = pattern[turn % 4]; // 如果是4则是1:1:1:1，如果是5则是3:3:2:2
                    for (int i = 0; i < insertPattern.length; i++) {
                        trainList.add(insertPattern[i]);
                    }
                }
            } else if (false) {
                // 生成需要的训练字符串，格式是随机混淆字符串+随机训练字符串+随机混淆字符串+随机训练字符串+...
                Random rand1 = new Random(1);
                for (int turn = 0; turn < 20; turn++) {
                    int nauty_num = rand1.nextInt(10);
                    for (int i = 0; i < nauty_num; i++) {
                        trainList.add(nauty[rand1.nextInt(nauty.length)]);
                    }
                    String[] insertPattern = pattern[rand1.nextInt(pattern.length)];
                    for (int i = 0; i < insertPattern.length; i++) {
                        trainList.add(insertPattern[i]);
                    }
                }
            } else {
                // 先读入所有字
                BufferedReader br = new BufferedReader(new FileReader(new File("data\\test4")));
                String line = null;
                while ((line = br.readLine()) != null) {
                    String[] words = line.trim().split(" ");
                    if (words.length > 1) {
                        for (String word : words) {
                            trainList.add(word);
                        }
                    }
                }
                br.close();
            }
            // 输出原始训练语句
            if (true) {
                trainList.forEach(w -> System.out.print(w + " "));
                System.out.println("");
            }
            // 将所有词放入word_count
            HashMap<String, Integer> word_count = new HashMap<String, Integer>();
            for (String word : trainList) {
                Integer countInteger = word_count.get(word);
                if (countInteger != null) {
                    word_count.put(word, countInteger + 1);
                } else {
                    word_count.put(word, 1);
                }
            }
            // 排序
            List<Entry<String, Integer>> word_count_sorted = word_count.entrySet()
                                                                       .stream()
                                                                       .sorted(Comparator.comparing(e -> -e.getValue()))
                                                                       .collect(Collectors.toList());
            // 输出word_count
            if (true) {
                word_count_sorted.forEach(e -> System.out.println(e.getKey() + ":" + e.getValue()));
            }
            // 根据词频给词编号
            HashMap<String, Integer> wordCode = new HashMap<String, Integer>();
            HashMap<Integer, String> codeWord = new HashMap<Integer, String>();
            int code = 0;
            for (Entry<String, Integer> e : word_count_sorted) {
                codeWord.put(code, e.getKey());
                wordCode.put(e.getKey(), code);
                code++;
            }
            // 输出编号
            if (true) {
                wordCode.entrySet()
                        .forEach(e -> System.out.println(e.getKey() + ":" + e.getValue()));
            }
            // 利用编号对原始语句进行重述
            int[] codeArr = new int[trainList.size()];
            for (int i = 0; i < trainList.size(); i++) {
                codeArr[i] = wordCode.get(trainList.get(i));
            }
            // 输出重述后的语句
            if (true) {
                for (int c : codeArr) {
                    System.out.print(c + " ");
                }
                System.out.println("");
            }
            // 按照编号构造共现矩阵
            int window = 3;
            double[][] occurMat = new double[wordCode.size()][wordCode.size()];
            for (int i = 0; i < codeArr.length; i++) {
                // 遍历目标单词左边的窗口内的单词
                for (int j = i - window; j >= 0 && j < i; j++) {
                    occurMat[codeArr[i]][codeArr[j]] += 1.0 / (i - j);
                    // 原算法这里对对称性做了选择，我们不做对对称性的选择要求
                    occurMat[codeArr[j]][codeArr[i]] = occurMat[codeArr[i]][codeArr[j]];
                    // 注意这样产生的共现矩阵容易在某些常用词上特别大，在原算法中并没有看到明显的归一化处理的操作，认为这是可以接受的
                }
            }
            // 原模型中有打乱共现矩阵的行排序的操作，这里省略
            // 打印共现矩阵
            DecimalFormat df = new DecimalFormat("#0.0000");
            if (true) {
                System.out.print(" \t");
                for (int i = 0; i < wordCode.size(); i++) {
                    System.out.print(i + "\t");
                }
                System.out.println("");
                for (int i = 0; i < wordCode.size(); i++) {
                    System.out.print(i + "\t");
                    for (int j = 0; j < wordCode.size(); j++) {
                        System.out.print(df.format(occurMat[i][j]) + "\t");
                    }
                    System.out.println("");
                }
            }
            // 单词数量为n，在原模型中每个单词需要构造两个向量，一个是出向量，一个是入向量
            // 我们这里进行简化，考虑出向量和入向量是一码事，这个和上面的对称性是一种取舍
            // 构造权重矩阵，取dim=2，有 W[w0, w1, bw]和梯度矩阵g[g0, g1, gb]
            int n = wordCode.size();
            int dim = 3; // 两个w一个b
            double[][] w = new double[n][dim];
            double[][] g = new double[n][dim];
            // 对w矩阵初始化随机数，g矩阵初始化为1.0
            Random rand2 = new Random(2);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < dim; j++) {
                    w[i][j] = rand2.nextDouble();
                    g[i][j] = 1.0;
                }
            }
            // 遍历共现矩阵，计算点积
            double min_thres = 1; // 共现矩阵的最小值，这里用这个值代替原有的CREC的存在性
            double x_max = 10.0; // 论文中非线性函数f(x)的转折点，原模型中以10进行训练
            // 这里采用均匀遍历的方法，替代过去结构体的乱序训练，这种替代不会破坏算法核心计算思想
            double cost = 1.0; // 损失值
            double lastcost = 2.0; // 上一次损失值
            // 加入shuffle的元素
            Random rand3 = new Random(3);
            ArrayList<Integer> seq1 = new ArrayList<Integer>();
            ArrayList<Integer> seq2 = new ArrayList<Integer>();
            for (int i = 0; i < n; i++) {
                seq1.add(i);
                seq2.add(i);
            }
            for (int epouch = 0; epouch < 100000000 && cost > 1e-20
                    && Math.abs((lastcost - cost) / cost) > 1e-11; epouch++) {
                lastcost = cost;
                cost = 0.0;
                // 计算过程进行乱序
                Collections.shuffle(seq1, rand3);
                Collections.shuffle(seq2, rand3);
                for (Integer i : seq1) {
                    for (Integer j : seq2) {
                        if (occurMat[i][j] > min_thres) {
                            double diff = 0;
                            // 计算wiwj+bi+bj
                            for (int d = 0; d < dim - 1; d++) {
                                diff += w[i][d] * w[j][d];
                            }
                            // 计算bi+bj
                            diff += w[i][dim - 1] + w[j][dim - 1];
                            // 添加log(X)
                            diff -= Math.log(occurMat[i][j]);
                            // 计算损失函数需要的f(X)
                            double f_diff = (occurMat[i][j] > x_max) ? diff
                                    : Math.pow(occurMat[i][j] / x_max, 0.75) * diff;
                            // 计算损失函数（）
                            cost += 0.5 * f_diff * diff;
                            // 更新值乘以学习率，原模型是0.05
                            f_diff *= 0.1;
                            // 更新w
                            for (int d = 0; d < dim - 1; d++) {
                                // 单词向量的学习率乘以梯度，注意这里i和j的顺序
                                double temp1 = f_diff * w[j][d];
                                double temp2 = f_diff * w[i][d];
                                w[i][d] -= temp1 / Math.sqrt(g[i][d]);
                                w[j][d] -= temp2 / Math.sqrt(g[j][d]);
                                g[i][d] += temp1 * temp1;
                                g[j][d] += temp2 * temp2;
                            }
                            // 更新b
                            w[i][dim - 1] -= f_diff / Math.sqrt(g[i][dim - 1]);
                            w[j][dim - 1] -= f_diff / Math.sqrt(g[j][dim - 1]);
                            f_diff *= f_diff;
                            g[i][dim - 1] += f_diff;
                            g[j][dim - 1] += f_diff;
                        }
                    }
                }
                if (epouch % 200000 == 0) {
                    System.out.println("第[" + epouch + "]次训练，损失值为=" + cost + "，损失率变动="
                            + ((lastcost - cost) / cost));
                }
            }
            // 输出最终的词向量矩阵w，注意w矩阵中的偏置项b不要
            if (true) {
                for (int i = 0; i < n; i++) {
                    StringBuffer sb = new StringBuffer();
                    sb.append(codeWord.get(i));
                    for (int d = 0; d < dim - 1; d++) {
                        sb.append("\t" + df.format(w[i][d]));
                    }
                    System.out.println(sb.toString());
                }
            }
            // 计算两两向量差的欧氏距离
            if (true) {
                Set<String> ignore = new HashSet<String>();
                for (String i : nauty) {
                    ignore.add(i);
                }
                System.out.println("向量之间的欧氏距离（忽略x、y点）");
                System.out.print("欧距\t");
                for (int i = 1; i < n; i++) {
                    for (int j = 1; j < n; j++) {
                        if (i != j && !ignore.contains(codeWord.get(i))
                                && !ignore.contains(codeWord.get(j))) {
                            System.out.print(codeWord.get(i) + codeWord.get(j) + "\t");
                        }
                    }
                }
                System.out.println("");
                // ij是行，pq是列
                for (int i = 1; i < n; i++) {
                    for (int j = 1; j < n; j++) {
                        if (i != j && !ignore.contains(codeWord.get(i))
                                && !ignore.contains(codeWord.get(j))) {
                            StringBuffer sb = new StringBuffer();
                            sb.append(codeWord.get(i) + codeWord.get(j));
                            for (int p = 1; p < n; p++) {
                                for (int q = 1; q < n; q++) {
                                    if (p != q && !ignore.contains(codeWord.get(p))
                                            && !ignore.contains(codeWord.get(q))) {
                                        double sum = 0;
                                        for (int d = 0; d < dim - 1; d++) {
                                            sum += ((w[i][d] - w[j][d]) - (w[p][d] - w[q][d]))
                                                    * ((w[i][d] - w[j][d]) - (w[p][d] - w[q][d]));
                                        }
                                        sb.append("\t" + df.format(Math.sqrt(sum)));
                                    }
                                }
                            }
                            System.out.println(sb.toString());
                        }
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
