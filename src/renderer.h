#ifndef RENDERER_H
#define RENDERER_H

#include "RT_struct.h"

class Camera {
private:
	Vec3 origin_;
	Vec3 lower_left_corner_;
	Vec3 horizontal_;
	Vec3 vertical_;
	Vec3 u_, v_, w_;
	double lens_radius_;

public:
	Camera() {}
	void init(Vec3 from, Vec3 lookat, Vec3 up, double fov, double aspect, double aperture, double dist_to_focus);

	Ray get_ray(double s, double t, my_rand &rnd) const {
//		Vec3 rd = Vec3::random_in_unit_disc(rnd) * (Vec3(lens_radius_, lens_radius_) + 5.0 * bias);
//		Vec3 rd = Vec3::random_in_unit_disc(rnd) * lens_radius_ + bias;
		Vec3 rd = Vec3::random_in_unit_disc(rnd) * lens_radius_;
		Vec3 offset = u_ * rd.x + v_ * rd.y;
		return Ray(origin_ + offset, lower_left_corner_ + horizontal_ * s + vertical_ * t - origin_ - offset);
	}
};

class IBL {
private:
	int w_ = 0, h_=0;
	double dw_ = 0.0, dh_=0.0;
	double *pImage_=nullptr;
public:
	IBL() { w_ = 0; h_ = 0; dw_ = 0.0; dh_ = 0.0; pImage_ = nullptr; }
	~IBL() { SAFE_DELETE_ARRAY(pImage_); }

	bool initialize(int w, int h, const float *p);

	inline Vec3 get(double u, double v) const { double *p = pImage_ + 3 * (w_*(int)(v*dh_) + (int)(u*dw_)); return Vec3(p[0], p[1], p[2]); }
//	inline Vec3 get(const Vec3 d) const { return get(0.5 * d.x + 0.5, 0.5*d.y + 0.5); }
	inline Vec3 get(const Vec3 d) const { return get(atan2(d.x, d.z) / (2.0*PI) + 0.5, acos(-d.y) / PI); }
};

class renderer{
private:
	int steps_;
	int WIDTH;
	int HEIGHT;
	Camera cam_;
	HitableList scene_;
	IBL ibl_;

	Vec3 raytrace(Ray r, int depth, my_rand &rnd)const;
public:
	renderer(int w, int h);
	~renderer();
	
	void setIBL(int width, int height, const float *image);

	void update(const double *src, double *dest, const double *normal_map)const;// なるべく早く出ること
	
	void copy(const double *src, double *dest)const;
	void median_filter(const double *src, double *dest) const;
	void get_luminance(const double *src, double *dest) const;
	void edge_detection(const double *src, double *dest) const;
	void gauss_blur_x(const double *src, double *dest) const;
	void gauss_blur_y(const double *src, double *dest) const;
	void compute_normal(const double *src, double *dest) const;
};
#endif // !RENDERER_H
