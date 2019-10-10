package cn.dendarii.model;

import java.util.Random;

import cn.dendarii.Mat;

public class BasicRnn {
    // w矩阵和b矩阵
    public double[][] wx;
    public double[][] wh;
    public double[][] wy;
    public double[] bh;
    public double[] by;
    // h和o状态
    public double[] h;
    public double[] y;
    /**
     * @param modelDim 模型内部的维度
     * @param outDim 模型输入和输出维度
     */
    public BasicRnn(int modelDim,
                    int outDim) {
        wx = new double[modelDim][outDim];
        wh = new double[modelDim][modelDim];
        wy = new double[modelDim][outDim];
        bh = new double[modelDim];
        by = new double[outDim];
        h = new double[modelDim];
        y = new double[outDim];
    }
    /**
     * @param initH 是否对h和o状态初始化
     */
    public BasicRnn(Random rand,
                    int modelDim,
                    int outDim,
                    boolean initH) {
        this(modelDim, outDim);
        // 初始化对应的矩阵
        Mat.apply(wx, x -> rand.nextDouble(), false);
        Mat.apply(wh, x -> rand.nextDouble(), false);
        Mat.apply(wy, x -> rand.nextDouble(), false);
        Mat.apply(bh, x -> rand.nextDouble(), false);
        Mat.apply(by, x -> rand.nextDouble(), false);
        if (initH) {
            Mat.apply(h, x -> rand.nextDouble(), false);
            Mat.normalize2d(h);
        }
    }
    /**
     * @param output 是否输出过程信息
     */
    public void forward(double[] x_t_in,
                        double[] h_prev,
                        boolean printPath) {
        // 计算h，h重新赋值
        h = Mat.add(Mat.mul(wh, h_prev), Mat.mul(wx, x_t_in), bh);
        Mat.apply(h, x -> Math.tanh(x), false);
        // 计算y
        y = Mat.add(Mat.mul(Mat.reverse(wy), h), by);
        Mat.softmax(y);
    }
    public double[] backward(double[] x_t_in,
                             double[] dy,
                             double[] h_prev,
                             BasicRnn drnn,
                             boolean printPath) {
        // 计算dwy
        double[][] dwy = Mat.mulx1d1d(h, dy);
        Mat.applyAdd(drnn.wy, dwy, false);
        // 计算dby
        double[] dby = dy;
        Mat.applyAdd(drnn.by, dby, false);
        // 计算dh
        double[] dh = Mat.mul(wy, dy);
        double[] dh_ = Mat.applyMul(dh, Mat.tanhg(h));
        // 计算dwh
        double[][] dwh = Mat.mulx1d1d(h_prev, dh_);
        Mat.applyAdd(drnn.wh, Mat.reverse(dwh), false);
        // 计算dwx
        double[][] dwx = Mat.mulx1d1d(x_t_in, dh_);
        Mat.applyAdd(drnn.wx, Mat.reverse(dwx), false);
        // 计算dbh
        double[] dbh = dh_;
        Mat.applyAdd(drnn.bh, dbh, false);
        // 计算dx
        double[] dx = new double[x_t_in.length];
        for (int d_rol = 0; d_rol < x_t_in.length; d_rol++) {
            for (int d_col = 0; d_col < h.length; d_col++) {
                dx[d_rol] = wx[d_col][d_rol] * dh[d_col];
            }
        }
        return dx;
    }
    public void update(BasicRnn drnn) {
        // 更新各个参数
        Mat.update(wy, drnn.wy);
        Mat.update(by, drnn.by);
        Mat.update(wh, drnn.wh);
        Mat.update(wx, drnn.wx);
        Mat.update(bh, drnn.bh);
    }
}
