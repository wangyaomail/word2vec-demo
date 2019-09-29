package cn.dendarii;

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
                              double[] dsrc,
                              double alpha) {
        for (int i = 0; i < src.length; i++) {
            src[i] -= alpha * dsrc[i];
        }
    }
    /**
     * 按照alpha更新第一个数组
     */
    public static void update(double[][] src,
                              double[][] dsrc,
                              double alpha) {
        for (int i = 0; i < src.length; i++) {
            for (int j = 0; j < src[0].length; j++) {
                src[i][j] -= alpha * dsrc[i][j];
            }
        }
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
        for (int i = 0; i < m.length; i++) {
            m[i] /= sum;
        }
    }
    /**
     * 归一化
     * 
     * @param byRow
     *            是否按行归一化
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
}
