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
	int w_, h_;
	double dw_, dh_;
	FrameBuffer *pImage_;
public:
	IBL() { w_ = 0; h_ = 0; dw_ = 0.0; dh_ = 0.0; pImage_ = nullptr; }
	~IBL() { SAFE_DELETE_ARRAY(pImage_); }

	bool initialize(int w, int h, const float *p);

	inline Color get(double u, double v) const { return pImage_->get((int)(u*dw_), (int)(v*dh_)); }
//	inline Color get(const Vec3 d) const { return get(0.5 * d.x + 0.5, 0.5*d.y + 0.5); }
	inline Color get(const Vec3 d) const { return get(atan2(d.x, d.z) / (2.0*PI) + 0.5, acos(-d.y) / PI); }// 上の式で読めるように変換したい…
};

class renderer{
private:
	int steps_;
	int WIDTH;
	int HEIGHT;
	Camera cam_;
	HitableList scene_;
	IBL ibl_;

	Color raytrace(Ray r, int depth, my_rand &rnd)const;
public:
	renderer(int w, int h);
	~renderer();
	
	void setIBL(int width, int height, const float *image);

	void update(const FrameBuffer *src, FrameBuffer *dest, const FrameBuffer *normal_map)const;// なるべく早く出ること
	
	static void copy(const FrameBuffer &src, FrameBuffer &dest);
	static void median_filter(const FrameBuffer &src, FrameBuffer &dest);
	static void get_luminance(const FrameBuffer &src, FrameBuffer &dest);
	static void edge_detection(const FrameBuffer &src, FrameBuffer &dest);
	static void gauss_blur_x(const FrameBuffer &src, FrameBuffer &dest);
	static void gauss_blur_y(const FrameBuffer &src, FrameBuffer &dest);
	static void compute_normal(const FrameBuffer &src, FrameBuffer &dest);
};
#endif // !RENDERER_H
