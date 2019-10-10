package cn.dendarii.model;

import java.util.Random;

import cn.dendarii.Mat;

public class Lstm {
    // 默认参数
    int dim = 3;
    int word_num = 3;
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
    public double[] c = new double[dim];
    public double[] c_ = new double[dim];
    public double[] h = new double[dim];
    public double[] o = new double[dim];
    public double[] y = new double[word_num];
    public Lstm(int dim,
                int word_num) {
        // 重新赋值
        this.dim = dim;
        this.word_num = word_num;
        wfh = new double[dim][dim];
        wfx = new double[dim][word_num];
        bf = new double[dim];
        wih = new double[dim][dim];
        wix = new double[dim][word_num];
        bi = new double[dim];
        wch = new double[dim][dim];
        wcx = new double[dim][word_num];
        bc = new double[dim];
        woh = new double[dim][dim];
        wox = new double[dim][word_num];
        bo = new double[dim];
        wy = new double[dim][word_num];
        by = new double[word_num];
        f = new double[dim];
        i = new double[dim];
        c = new double[dim];
        c_ = new double[dim];
        h = new double[dim];
        o = new double[dim];
        y = new double[word_num];
    }
    public Lstm(Random rand,
                int dim,
                int word_num) {
        this(dim, word_num);
        // 初始化对应的矩阵
        Mat.apply(wfh, x -> rand.nextDouble(), false);
        Mat.apply(wfx, x -> rand.nextDouble(), false);
        Mat.apply(bf, x -> rand.nextDouble(), false);
        Mat.apply(wih, x -> rand.nextDouble(), false);
        Mat.apply(wix, x -> rand.nextDouble(), false);
        Mat.apply(bi, x -> rand.nextDouble(), false);
        Mat.apply(wch, x -> rand.nextDouble(), false);
        Mat.apply(wcx, x -> rand.nextDouble(), false);
        Mat.apply(bc, x -> rand.nextDouble(), false);
        Mat.apply(woh, x -> rand.nextDouble(), false);
        Mat.apply(wox, x -> rand.nextDouble(), false);
        Mat.apply(bo, x -> rand.nextDouble(), false);
        Mat.apply(wy, x -> rand.nextDouble(), false);
        Mat.apply(by, x -> rand.nextDouble(), false);
        // Mat.apply(c, x -> rand.nextDouble(), false);
        // Mat.apply(h, x -> rand.nextDouble(), false);
        // Mat.normalize2d(c);
        // Mat.normalize2d(h);
    }
    public void forward(double[] x_t_in,
                        double[] h_prev,
                        double[] c_prev,
                        boolean output) {
        // 计算f
        f = Mat.add(Mat.mul(wfh, h_prev), Mat.mul(wfx, x_t_in), bf);
        Mat.apply(f, x -> Mat.sigmoid(x), false);
        // 计算i
        i = Mat.add(Mat.mul(wih, h_prev), Mat.mul(wix, x_t_in), bi);
        Mat.apply(i, x -> Mat.sigmoid(x), false);
        // 计算c_
        c_ = Mat.add(Mat.mul(wch, h_prev), Mat.mul(wcx, x_t_in), bc);
        Mat.apply(c_, x -> Math.tanh(x), false);
        // 计算c
        c = Mat.add(Mat.applyMul(f, c_prev), Mat.applyMul(i, c_));
        // 计算o
        o = Mat.add(Mat.mul(woh, h_prev), Mat.mul(wox, x_t_in), bo);
        Mat.apply(o, x -> Mat.sigmoid(x), false);
        // 计算h
        h = Mat.apply(c, x -> Math.tanh(x), true);
        h = Mat.applyMul(o, h);
        // 计算y
        y = Mat.add(Mat.mul(Mat.reverse(wy), h), by);
        Mat.softmax(y);
    }
    /**
     * 由dy提供梯度
     */
    public double[] backward(double[] x_t_in,
                             double[] dy,
                             double[] h_prev,
                             double[] c_prev,
                             Lstm dlstm,
                             boolean returndy,
                             boolean output) {
        // 1.计算dwy
        double[][] dwy = Mat.mulx1d1d(h, dy);
        Mat.applyAdd(dlstm.wy, dwy, false);
        Mat.applyAdd(dlstm.by, dy, false);
        // 2.计算dh
        double[] dh = Mat.mul(wy, dy);
        return backwardByDh(x_t_in, dh, h_prev, c_prev, dlstm, returndy, output);
    }
    /**
     * 
     * @param x_t_in
     * @param dh
     * @param h_prev
     * @param c_prev
     * @param dlstm
     * @param returndy
     *            true：返回dy，false：返回dh
     * @param output
     *            是否输出需要的过程信息
     * @return
     */
    public double[] backwardByDh(double[] x_t_in,
                                 double[] dh,
                                 double[] h_prev,
                                 double[] c_prev,
                                 Lstm dlstm,
                                 boolean returndy,
                                 boolean output) {
        // 计算do，因关键字原因带下划线
        double[] do_ = Mat.applyMul(c, dh);
        do_ = Mat.applyMul(do_, Mat.sigmoidg(o));
        Mat.applyAdd(dlstm.bo, do_, false);
        // 计算dwoh
        double[][] dwoh = Mat.mulx1d1d(h_prev, do_);
        Mat.applyAdd(dlstm.woh, Mat.reverse(dwoh), false);
        // 计算dwox
        double[][] dwox = Mat.mulx1d1d(x_t_in, do_);
        Mat.applyAdd(dlstm.wox, Mat.reverse(dwox), false);
        // 3.计算dc
        double[] dc = Mat.applyMul(dh, o);
        dc = Mat.applyMul(dc, Mat.tanhg(c));
        // 4.计算dc_
        double[] dc_ = Mat.applyMul(dc, i);
        dc_ = Mat.applyMul(dc_, Mat.tanhg(c_));
        Mat.applyAdd(dlstm.bc, dc_, false);
        // 计算dwch
        double[][] dwch = Mat.mulx1d1d(h_prev, dc_);
        Mat.applyAdd(dlstm.wch, Mat.reverse(dwch), false);
        // 计算dwcx
        double[][] dwcx = Mat.mulx1d1d(x_t_in, dc_);
        Mat.applyAdd(dlstm.wcx, Mat.reverse(dwcx), false);
        // 5.计算df
        double[] df = Mat.applyMul(dc, c_prev);
        df = Mat.applyMul(df, Mat.sigmoidg(f));
        Mat.applyAdd(dlstm.bf, df, false);
        // 计算dwfh
        double[][] dwfh = Mat.mulx1d1d(h_prev, df);
        Mat.applyAdd(dlstm.wfh, Mat.reverse(dwfh), false);
        // 计算dwfx
        double[][] dwfx = Mat.mulx1d1d(x_t_in, df);
        Mat.applyAdd(dlstm.wfx, Mat.reverse(dwfx), false);
        // 6.计算di
        double[] di = Mat.applyMul(dc, c_);
        di = Mat.applyMul(di, Mat.sigmoidg(i));
        Mat.applyAdd(dlstm.bi, di, false);
        // 计算dwfh
        double[][] dwih = Mat.mulx1d1d(h_prev, di);
        Mat.applyAdd(dlstm.wih, Mat.reverse(dwih), false);
        // 计算dwfx
        double[][] dwix = Mat.mulx1d1d(x_t_in, di);
        Mat.applyAdd(dlstm.wix, Mat.reverse(dwix), false);
        if (returndy) {
            // 计算上一层的dx，取f、i、c_、o的前向x误差平均值
            double[] dx = new double[word_num];
            for (int d_rol = 0; d_rol < word_num; d_rol++) {
                for (int d_col = 0; d_col < dim; d_col++) {
                    dx[d_rol] += wfx[d_col][d_rol] * df[d_col];
                    dx[d_rol] += wix[d_col][d_rol] * di[d_col];
                    dx[d_rol] += wcx[d_col][d_rol] * dc_[d_col];
                    dx[d_rol] += wox[d_col][d_rol] * do_[d_col];
                }
                dx[d_rol] = dx[d_rol] / 4;
            }
            return dx;
        } else {
            // 返回dh
            double[] return_dh = new double[dim];
            for (int d_rol = 0; d_rol < dim; d_rol++) {
                for (int d_col = 0; d_col < dim; d_col++) {
                    return_dh[d_rol] += wfh[d_col][d_rol] * df[d_col];
                    return_dh[d_rol] += wih[d_col][d_rol] * di[d_col];
                    return_dh[d_rol] += wch[d_col][d_rol] * dc_[d_col];
                    return_dh[d_rol] += woh[d_col][d_rol] * do_[d_col];
                }
                return_dh[d_rol] = return_dh[d_rol] / 4;
            }
            return return_dh;
        }
        // // 尝试归一化h和c
        // Mat.normalize2d(c);
        // Mat.normalize2d(h);
    }
    // 这里不需要区分y还是c在梯度中的影响
    public void update(Lstm dlstm) {
        // 更新各个参数
        Mat.update(wy, dlstm.wy);
        Mat.update(by, dlstm.by);
        Mat.update(woh, dlstm.woh);
        Mat.update(wox, dlstm.wox);
        Mat.update(bo, dlstm.bo);
        Mat.update(wch, dlstm.wch);
        Mat.update(wcx, dlstm.wcx);
        Mat.update(bc, dlstm.bc);
        Mat.update(wfh, dlstm.wfh);
        Mat.update(wfx, dlstm.wfx);
        Mat.update(bf, dlstm.bf);
        Mat.update(wih, dlstm.wih);
        Mat.update(wix, dlstm.wix);
        Mat.update(bi, dlstm.bi);
    }
}
