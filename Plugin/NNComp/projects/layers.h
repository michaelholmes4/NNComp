#pragma once
#include "Eigen/Dense"

/*
// ---- Fast Approximations ---
double exp_fast(double x) {
    auto numerator = 1680 + x * (840 + x * (180 + x * (20 + x)));
    auto denominator = 1680 + x *(-840 + x * (180 + x * (-20 + x)));
    return numerator / denominator;
}

double tanh_fast (const double x) {
    auto x2 = x * x;
    auto numerator = x * (135135 + x2 * (17325 + x2 * (378 + x2)));
    auto denominator = 135135 + x2 * (62370 + x2 * (3150 + 28 * x2));
    return numerator / denominator;
}
 */


/**
 * Sigmoid activation function
 */
double Sigmoid(const double z) {
  return 1.0 / (1.0 + exp(-z));
  //return 1.0 / (1.0 + exp_fast(-z)); //Fast Approx.
}

/**
 * Defines a single lstm layer.
 * T - Type
 * h - hidden size
 * d - input size
 */
template<typename T, int h, int d>
class LstmLayer {
public:
  LstmLayer(){}
  
  void apply_layer(Eigen::Vector<T, d> x) {
    xt = x;
    it = (Wii * xt + bii + Whi * htn1 + bhi).unaryExpr(std::ref(Sigmoid));
    ft = (Wif * xt + bif + Whf * htn1 + bhf).unaryExpr(std::ref(Sigmoid));
    gt = (Wig * xt + big + Whg * htn1 + bhg).array().tanh();
    //gt = (Wig * xt + big + Whg * htn1 + bhg).unaryExpr(std::ref(tanh_fast)); //Fast Approx.
    ot = (Wio * xt + bio + Who * htn1 + bho).unaryExpr(std::ref(Sigmoid));
    ct = ft.cwiseProduct(ctn1) + it.cwiseProduct(gt);
    ht = ct.array().tanh();
    //ht = ct.unaryExpr(std::ref(tanh_fast)); //Fast Approx.
    ht = ot.cwiseProduct(ht);
    htn1 = ht;
    ctn1 = ct;
  }
  
  void apply_layer(T x) {
    xt = xt.Constant(x);
    it = (Wii * xt + bii + Whi * htn1 + bhi).unaryExpr(std::ref(Sigmoid));
    ft = (Wif * xt + bif + Whf * htn1 + bhf).unaryExpr(std::ref(Sigmoid));
    gt = (Wig * xt + big + Whg * htn1 + bhg).array().tanh();
    //gt = (Wig * xt + big + Whg * htn1 + bhg).unaryExpr(std::ref(tanh_fast)); //Fast Approx.
    ot = (Wio * xt + bio + Who * htn1 + bho).unaryExpr(std::ref(Sigmoid));
    ct = ft.cwiseProduct(ctn1) + it.cwiseProduct(gt);
    ht = ct.array().tanh();
    //ht = ct.unaryExpr(std::ref(tanh_fast)); //Fast Approx.
    ht = ot.cwiseProduct(ht);
    htn1 = ht;
    ctn1 = ct;
  }
  
  Eigen::Matrix<T, h, d> Wii;
  Eigen::Matrix<T, h, d> Wif;
  Eigen::Matrix<T, h, d> Wig;
  Eigen::Matrix<T, h, d> Wio;

  Eigen::Matrix<T, h, h> Whi;
  Eigen::Matrix<T, h, h> Whf;
  Eigen::Matrix<T, h, h> Whg;
  Eigen::Matrix<T, h, h> Who;

  Eigen::Vector<T, h> bii;
  Eigen::Vector<T, h> bif;
  Eigen::Vector<T, h> big;
  Eigen::Vector<T, h> bio;

  Eigen::Vector<T, h> bhi;
  Eigen::Vector<T, h> bhf;
  Eigen::Vector<T, h> bhg;
  Eigen::Vector<T, h> bho;
  
  Eigen::Vector<T, h> it;
  Eigen::Vector<T, h> ft;
  Eigen::Vector<T, h> gt;
  Eigen::Vector<T, h> ot;
  Eigen::Vector<T, h> ct;
  Eigen::Vector<T, h> ht;
  
  Eigen::Vector<T, d> xt;
  
  Eigen::Vector<T, h> ctn1;
  Eigen::Vector<T, h> htn1;
};

/**
 * Defines a single lstm layer.
 * T - Type
 * h - hidden size
 * d - input size
 */
template<typename T, int h, int d>
class GruLayer {
public:
  GruLayer(){}
  
  void apply_layer(Eigen::Vector<T, d> x) {
    xt = x;
    rt = (Wir * xt + bir + Whr * htn1 + bhr).unaryExpr(std::ref(Sigmoid));
    zt = (Wiz * xt + biz + Whz * htn1 + bhz).unaryExpr(std::ref(Sigmoid));
    nt = (Win * xt + bin + rt.cwiseProduct(Whn * htn1 + bhn)).array().tanh();
    //nt = (Win * xt + bin + rt.cwiseProduct(Whn * htn1 + bhn)).unaryExpr(std::ref(tanh_fast)); //Fast Approx.
    ht = (ht.Constant(1) - zt).cwiseProduct(nt) + zt.cwiseProduct(htn1);
    htn1 = ht;
  }
  
  void apply_layer(T x) {
    xt = xt.Constant(x);
    rt = (Wir * xt + bir + Whr * htn1 + bhr).unaryExpr(std::ref(Sigmoid));
    zt = (Wiz * xt + biz + Whz * htn1 + bhz).unaryExpr(std::ref(Sigmoid));
    nt = (Win * xt + bin + rt.cwiseProduct(Whn * htn1 + bhn)).array().tanh();
    //nt = (Win * xt + bin + rt.cwiseProduct(Whn * htn1 + bhn)).unaryExpr(std::ref(tanh_fast)); //Fast Approx.
    ht = (ht.Constant(1) - zt).cwiseProduct(nt) + zt.cwiseProduct(htn1);
    htn1 = ht;
  }
  
  
  Eigen::Matrix<T, h, d> Wir;
  Eigen::Matrix<T, h, d> Wiz;
  Eigen::Matrix<T, h, d> Win;

  Eigen::Matrix<T, h, h> Whr;
  Eigen::Matrix<T, h, h> Whz;
  Eigen::Matrix<T, h, h> Whn;

  Eigen::Vector<T, h> bir;
  Eigen::Vector<T, h> biz;
  Eigen::Vector<T, h> bin;

  Eigen::Vector<T, h> bhr;
  Eigen::Vector<T, h> bhz;
  Eigen::Vector<T, h> bhn;
  
  Eigen::Vector<T, h> rt;
  Eigen::Vector<T, h> zt;
  Eigen::Vector<T, h> nt;
  Eigen::Vector<T, h> ht;
  
  Eigen::Vector<T, d> xt;
  
  Eigen::Vector<T, h> htn1;
};

/**
 * Defines a single lstm layer.
 * T - Type
 * h - hidden size
 * d - input size
 */
template<typename T, int h, int d>
class RnnLayer {
public:
  RnnLayer(){}
  
  void apply_layer(Eigen::Vector<T, d> x) {
    xt = x;
    ht = (Wih * xt + bih + Whh * htn1 + bhh).array().tanh();
    //ht = (Wih * xt + bih + Whh * htn1 + bhh).unaryExpr(std::ref(tanh_fast)); //Fast Approx.
    htn1 = ht;
  }
  
  void apply_layer(T x) {
    xt = xt.Constant(x);
    ht = (Wih * xt + bih + Whh * htn1 + bhh).array().tanh();
    //ht = (Wih * xt + bih + Whh * htn1 + bhh).unaryExpr(std::ref(tanh_fast)); //Fast Approx.
    htn1 = ht;
  }
  Eigen::Matrix<T, h, d> Wih;
  Eigen::Matrix<T, h, h> Whh;
  Eigen::Vector<T, h> bih;
  Eigen::Vector<T, h> bhh;
  Eigen::Vector<T, h> ht;
  Eigen::Vector<T, d> xt;
  Eigen::Vector<T, h> htn1;
};

/**
 * Defines a fully connected layer
 * T - Type
 * h - hidden size
 * d - input size
 */
template<typename T, int h, int d>
class FccLayer {
public:
  FccLayer(){}
  T apply_layer(Eigen::Vector<T, h> x) {
    xt = x;
    y = (xt.transpose() * A + b).array().tanh()(0);
    return y;
    //return (x.transpose() * A + b).array().unaryExpr(std::ref(tanh_fast))(0); //Fast Approx.
  }
  
  Eigen::Matrix<T, h, d> A;
  Eigen::Vector<T, d> b;
  Eigen::Vector<T, h> xt;
  T y;
  
};
