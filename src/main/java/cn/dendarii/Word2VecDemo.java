package cn.dendarii;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeSet;
import java.util.stream.Collectors;

@SuppressWarnings("unused")
public class Word2VecDemo {
    public static void main(String[] args) {
        try {
            // 先读入所有字
            List<String[]> trainList = new ArrayList<String[]>();
            BufferedReader br = new BufferedReader(new FileReader(new File("data\\test2")));
            String line = null;
            while ((line = br.readLine()) != null) {
                String[] words = line.trim().split("");
                if (words.length > 1) {
                    trainList.add(words);
                }
            }
            br.close();
            // 将所有词放入word_count
            HashMap<String, Integer> word_count = new HashMap<String, Integer>();
            int count = 0;
            for (String[] words : trainList) {
                for (int i = 0; i < words.length; i++) {
                    Integer countInteger = word_count.get(words[i]);
                    if (countInteger != null) {
                        word_count.put(words[i], countInteger + 1);
                    } else {
                        word_count.put(words[i], 1);
                    }
                }
                count += words.length;
            }
            // 输出word_count
            if (false) {
                word_count.entrySet()
                          .forEach(e -> System.out.println(e.getKey() + ":" + e.getValue()));
            }
            // 准备huffman树用到的node
            int dim = 2; // 向量深度
            Random random = new Random();
            class Node implements Comparable<Node> {
                public String word;
                public Integer count;
                public Node left;
                public Node right;
                public int code;
                public Node father;
                public List<Node> pathNodes;
                public int[] path;
                // 下面这三个参数只和训练有关
                public int trainCount = 0; // 这个词被训练了多少次
                public double[] k = new double[dim]; // 对隐藏层来说k是分类器，对节点层来说k无效
                public double[] x = new double[dim]; // 对节点层来说x是词向量，对隐藏层来说x无效
                public Node(String word,
                            Integer count,
                            Node left,
                            Node right) {
                    this.word = word;
                    this.count = count;
                    if (left != null) {
                        this.left = left;
                        left.father = this;
                        left.code = 0;
                    }
                    if (right != null) {
                        this.right = right;
                        right.father = this;
                        right.code = 1;
                    }
                    if (word != null) {// 如果是词节点，需要对x进行初值随机化
                        for (int i = 0; i < dim; i++) {
                            x[i] = random.nextDouble() - 0.5;
                        }
                    }
                }
                public int compareTo(Node o) {
                    return this.count > o.count ? 1 : -1;
                }
            }
            List<Node> wordNodes = word_count.entrySet()
                                             .stream()
                                             .map(e -> new Node(e.getKey(),
                                                                e.getValue(),
                                                                null,
                                                                null))
                                             .collect(Collectors.toList());
            Map<String, Node> nodeMap = wordNodes.stream()
                                                 .collect(Collectors.toMap(n -> n.word, n -> n));
            TreeSet<Node> allNodes = new TreeSet<Node>(wordNodes);
            // 构建huffman树
            while (allNodes.size() > 1) {
                Node left = allNodes.pollFirst();
                Node right = allNodes.pollFirst();
                allNodes.add(new Node(null, left.count + right.code, left, right));
            }
            // 构建每个词的编码
            for (Node word : wordNodes) {
                word.pathNodes = new ArrayList<Node>();
                Node node = word;
                word.pathNodes.add(node);
                while ((node = node.father) != null) {
                    word.pathNodes.add(node);
                }
                Collections.reverse(word.pathNodes);
                word.path = new int[word.pathNodes.size()];
                for (int i = 0; i < word.pathNodes.size(); i++) {
                    word.path[i] = word.pathNodes.get(i).code;
                }
            }
            if (true) {
                // 输出每个词的huffman编码
                for (Node word : wordNodes) {
                    StringBuffer sb = new StringBuffer();
                    for (int i = 0; i < word.path.length; i++) {
                        sb.append(word.path[i]);
                    }
                    System.out.println(word.word + "：\t" + sb.toString());
                }
            }
            // 降采样简化策略，每次训练一句话中训练次数最少的词
            for (int epouch = 0; epouch < 1000000; epouch++) {
                for (String[] toks : trainList) {
                    int pos = 0;
                    Node minNode = nodeMap.get(toks[0]);
                    for (int i = 1; i < toks.length; i++) {
                        if (minNode.trainCount > nodeMap.get(toks[i]).trainCount) {
                            pos = i;
                            minNode = nodeMap.get(toks[i]);
                        }
                    }
                    // 使用skipGram训练模型
                    int win = 2; // 窗口大小
                    for (int i = pos - win; i <= pos + win; i++) {
                        if (i < 0 || i >= toks.length || i == pos) {
                            continue;
                        }
                        double[] dx = new double[dim];// 误差项
                        // 遍历路径节点
                        for (Node pathNode : minNode.pathNodes) {
                            double f = 0;
                            // 让路径节点的k和窗口节点的x值相乘
                            for (int j = 0; j < dim; j++) {
                                f += pathNode.k[j] * minNode.x[j];
                            }
                            // 计算sigmoid分类值
                            f = 1 - 1 / (1 + Math.exp(f));
                            // 计算更新的梯度，交叉熵-1即是梯度
                            double g = -(f + pathNode.code - 1) * 0.025;
                            // 积累误差项
                            for (int j = 0; j < dim; j++) {
                                dx[j] += g * pathNode.k[j];
                            }
                            // 更新路径节点k
                            for (int j = 0; j < dim; j++) {
                                pathNode.k[j] += g * minNode.x[j];
                            }
                        }
                        // 更新窗口内词节点的累计误差
                        for (int j = 0; j < dim; j++) {
                            minNode.x[j] += dx[j];
                        }
                    }
                }
            }
            // 输出训练结果
            DecimalFormat df = new DecimalFormat("#0.00");
            for (Node word : wordNodes) {
                StringBuffer sb = new StringBuffer();
                for (int i = 0; i < word.x.length; i++) {
                    sb.append(df.format(word.x[i])).append(",");
                }
                System.out.println(word.word + "：\t" + sb.toString());
            }
            // 输出欧氏距离
            for (Node word1 : wordNodes) {
                System.out.print(word1.word + "\t");
                for (Node word2 : wordNodes) {
                    // System.out.println(word1.word + "-" + word2.word + "："
                    // + df.format(euclideanDistance(word1.x, word2.x)));
                    System.out.print(df.format(euclideanDistance(word1.x, word2.x)) + "\t");
                }
                System.out.println("");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    public static double euclideanDistance(double[] x,
                                           double[] y) {
        if (x == null || y == null || x.length != y.length) {
            return -1;
        } else {
            double sum = 0;
            for (int i = 0; i < x.length; i++) {
                sum += (x[i] - y[i]) * (x[i] - y[i]);
            }
            return Math.sqrt(sum);
        }
    }
}
