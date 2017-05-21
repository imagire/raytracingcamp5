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

	Ray get_ray(double s, double t) {
		Vec3 rd = Vec3::random_in_unit_disc() * lens_radius_;
		Vec3 offset = u_ * rd.x + v_ * rd.y;
		return Ray(origin_ + offset, lower_left_corner_ + horizontal_ * s + vertical_ * t - origin_ - offset);
	}
};


class renderer{
private:
	int steps_;
	int WIDTH;
	int HEIGHT;
	my_rand rand_;
	Camera cam_;
	HitableList scene_;

	Vec3 raytrace(Ray r, int depth);
	Vec3 color(double u, double v);
public:
	renderer(int w, int h);
	~renderer();
	
	int update(const double *src, double *dest);// �ɗ͑��������邱��
};

#endif // !RENDERER_H