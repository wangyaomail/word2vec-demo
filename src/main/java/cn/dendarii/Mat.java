package cn.dendarii;

import java.text.DecimalFormat;
import java.util.function.BiFunction;
import java.util.function.UnaryOperator;

/**
 * 这一函数仅适用于本工程的算法，为保证纯粹性不作任何异常判断和捕获
 */
public class Mat {
    /**
     * 计算向量长度
     */
    public static double vectorLength(double[] vector) {
        double sum = 0;
        for (int i = 0; i < vector.length; i++) {
            sum += vector[i] * vector[i];
        }
        return Math.sqrt(sum);
    }
    /**
     * 计算向量长度
     */
    public static double vectorLength(double[][] vector) {
        double sum = 0;
        for (int i = 0; i < vector.length; i++) {
            for (int j = 0; j < vector[0].length; j++) {
                sum += vector[i][j] * vector[i][j];
            }
        }
        return Math.sqrt(sum);
    }
    /**
     * 按照alpha更新第一个数组
     */
    public static void update(double[] src,
                              double[] dsrc) {
        update(src, dsrc, 0.05, 10);
    }
    public static void update(double[] src,
                              double[] dsrc,
                              double alpha,
                              double clip) {
        for (int i = 0; i < src.length; i++) {
            src[i] -= alpha * (dsrc[i] > clip ? clip : dsrc[i]);
        }
    }
    /**
     * 按照alpha更新第一个数组，默认alpha=0.05，clip=10
     */
    public static void update(double[][] src,
                              double[][] dsrc) {
        update(src, dsrc, 0.05, 10);
    }
    public static void update(double[][] src,
                              double[][] dsrc,
                              double alpha,
                              double clip) {
        for (int i = 0; i < src.length; i++) {
            for (int j = 0; j < src[0].length; j++) {
                src[i][j] -= alpha * (dsrc[i][j] > clip ? clip : dsrc[i][j]);
            }
        }
    }
    // 对矩阵每一项应用一个函数
    public static double[] apply(double[] a,
                                 UnaryOperator<Double> f,
                                 boolean newSet) {
        double[] y = newSet ? new double[a.length] : a;
        for (int i = 0; i < a.length; i++) {
            y[i] = f.apply(a[i]);
        }
        return y;
    }
    // 对矩阵每一项应用一个函数
    public static double[][] apply(double[][] a,
                                   UnaryOperator<Double> f,
                                   boolean newSet) {
        double[][] y = newSet ? new double[a.length][a[0].length] : a;
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                y[i][j] = f.apply(a[i][j]);
            }
        }
        return y;
    }
    // 对两个矩阵按位应用一个函数
    public static double[] apply(double[] a,
                                 double[] b,
                                 BiFunction<Double, Double, Double> f) {
        double[] y = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            y[i] = f.apply(a[i], b[i]);
        }
        return y;
    }
    /**
     * 按位加
     */
    public static double[] applyAdd(double[] a,
                                    double[] b) {
        double[] y = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            y[i] = a[i] + b[i];
        }
        return y;
    }
    /**
     * 按位减
     */
    public static double[] applyMinus(double[] a,
                                      double[] b) {
        double[] y = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            y[i] = a[i] - b[i];
        }
        return y;
    }
    /**
     * 按位乘
     */
    public static double[] applyMul(double[] a,
                                    double[] b) {
        double[] y = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            y[i] = a[i] * b[i];
        }
        return y;
    }
    // 对两个矩阵按位应用一个函数
    public static double[][] apply(double[][] a,
                                   double[][] b,
                                   BiFunction<Double, Double, Double> f) {
        double[][] y = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                y[i][j] = f.apply(a[i][j], b[i][j]);
            }
        }
        return y;
    }
    /**
     * 按位乘
     */
    public static double[][] applyMul(double[][] a,
                                      double[][] b) {
        double[][] y = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                y[i][j] = a[i][j] * b[i][j];
            }
        }
        return y;
    }
    /**
     * 归一化
     */
    public static void normalize1d(double[] m) {
        double sum = 0;
        for (int i = 0; i < m.length; i++) {
            sum += m[i];
        }
        for (int i = 0; i < m.length; i++) {
            m[i] /= sum;
        }
    }
    /**
     * 按照向量长度归一化
     */
    public static void normalize2d(double[] m) {
        double sum = 0;
        for (int i = 0; i < m.length; i++) {
            sum += m[i] * m[i];
        }
        sum = Math.sqrt(sum);
        for (int i = 0; i < m.length; i++) {
            m[i] /= sum;
        }
    }
    /**
     * 归一化
     */
    public static void normalize(double[][] m,
                                 boolean byRow) {
        if (byRow) {
            for (int i = 0; i < m.length; i++) {
                double sum = 0;
                for (int j = 0; j < m[0].length; j++) {
                    sum += m[i][j];
                }
                for (int j = 0; j < m[0].length; j++) {
                    m[i][j] /= sum;
                }
            }
        } else {
            double sum = 0;
            for (int i = 0; i < m.length; i++) {
                for (int j = 0; j < m[0].length; j++) {
                    sum += m[i][j];
                }
            }
            for (int i = 0; i < m.length; i++) {
                for (int j = 0; j < m[0].length; j++) {
                    m[i][j] /= sum;
                }
            }
        }
    }
    /**
     * sigmoid
     */
    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }
    /**
     * sigmoid导数
     */
    public static double[] sigmoidg(double[] x) {
        double[] y = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = (1 - x[i]) * x[i];
        }
        return y;
    }
    /**
     * softmax
     */
    public static void softmax(double[] x) {
        double sum = 0;
        for (int i = 0; i < x.length; i++) {
            x[i] = Math.exp(x[i]);
            sum += x[i];
        }
        for (int i = 0; i < x.length; i++) {
            x[i] = x[i] / sum;
        }
    }
    /**
     * tanh导数
     */
    public static double[] tanhg(double[] x) {
        double[] y = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = 1 - x[i] * x[i];
        }
        return y;
    }
    /**
     * 矩阵转置，不改变源矩阵
     */
    public static double[][] reverse(double[][] x) {
        double[][] y = new double[x[0].length][x.length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                y[j][i] = x[i][j];
            }
        }
        return y;
    }
    /**
     * 1x1矩阵乘法
     */
    public static double mul1d1d(double[] a,
                                 double[] b) {
        double y = 0;
        for (int i = 0; i < a.length; i++) {
            y += a[i] * b[i];
        }
        return y;
    }
    /**
     * 1x1矩阵乘法
     */
    public static double[][] mulx1d1d(double[] a,
                                      double[] b) {
        double[][] y = new double[a.length][b.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b.length; j++) {
                y[i][j] = a[i] * b[j];
            }
        }
        return y;
    }
    /**
     * 1x1矩阵加法
     */
    public static double[] add(double[]... x) {
        double[] y = new double[x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                y[j] += x[i][j];
            }
        }
        return y;
    }
    /**
     * 2x2矩阵加法
     */
    public static double[][] add(double[][]... x) {
        double[][] y = new double[x[0].length][x[1].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                for (int k = 0; k < x[0][0].length; k++) {
                    y[j][k] += x[i][j][k];
                }
            }
        }
        return y;
    }
    /**
     * 2x1矩阵乘法
     */
    public static double[] mul(double[][] a,
                               double[] b) {
        double[] y = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b.length; j++) {
                y[i] += a[i][j] * b[j];
            }
        }
        return y;
    }
    /**
     * 打印出向量值
     */
    public static String vectorString(String name,
                                      DecimalFormat df,
                                      double[] x) {
        StringBuilder sb = new StringBuilder();
        sb.append(name);
        for (int i = 0; i < x.length; i++) {
            sb.append("\t").append(df.format(x[i]));
        }
        return sb.toString();
    }
}
