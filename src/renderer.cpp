#include <cfloat>
#include <time.h>
#include <omp.h>
#include "renderer.h"

#define PI 3.141592653589793826433

void Camera::init(Vec3 from, Vec3 lookat, Vec3 up, double fov, double aspect, double aperture, double focus_dist)
{
	lens_radius_ = aperture / 2;
	double theta = fov * PI / 180;
	double half_height = tan(theta / 2);
	double half_width = aspect * half_height;
	origin_ = from;
	Vec3 d = from - lookat;
	w_ = d.normalize();
	u_ = cross(up, w_).normalize();
	v_ = cross(w_, u_);
	lower_left_corner_ = origin_ - u_ * (half_width * focus_dist) - v_ * (half_height * focus_dist) - w_ * focus_dist;
	horizontal_ = u_ * (2.0 * half_width * focus_dist);
	vertical_ = v_ * (2.0 * half_height * focus_dist);
}


renderer::renderer(int w, int h)
	:steps_(0)
{
	WIDTH = w;
	HEIGHT = h;

	// camera
	Vec3 from(13, 3, 3);
	Vec3 lookat(0, 0.5, 0);
	Vec3 up(0, 1, 0);
	double fov = 20.0;
	double aspect = (double)WIDTH / (double)HEIGHT;
	double dist_to_focus = 10.0;
	double aperture = 0.1;

	cam_.init(from, lookat, up, fov, aspect, aperture, dist_to_focus);

	// scene
	double R = cos(PI / 4);
	Material *p = new Lambertian(Vec3(0.5, 0.5, 0.5));
	scene_.Append(new Sphere(Vec3(0, -1000, 0), 1000, p));
	scene_.Append(new Sphere(Vec3(0, 1, 0), 1.0, new Dielectric(1.5)));
	scene_.Append(new Sphere(Vec3(-4, 1, 0), 1.0, new Lambertian(Vec3(0.4, 0.2, 0.1))));
	scene_.Append(new Sphere(Vec3(4, 1, 0), 1.0, new Metal(Vec3(0.7, 0.6, 0.5), 0.0)));
}

renderer::~renderer()
{
}

Vec3 renderer::raytrace(Ray r, int depth, my_rand &rnd)const
{
	// for debugging
//	return Vec3(0,0,0);
//	return Vec3(rand_.get(), rand_.get(), rand_.get());

	HitRecord rec;
	if (scene_.hit(r, 0.001, DBL_MAX, rec)) {
		Ray scattered;
		Vec3 attenuation;
		if (depth < 50 && rec.mat_ptr->scatter(r, rec, attenuation, scattered, rnd)) {
			return attenuation * raytrace(scattered, depth + 1, rnd);
		}
		else {
			return Vec3(0, 0, 0);
		}
	}
	else {
		Vec3 unit_direction = r.direction().normalize();
		double t = 0.5*(unit_direction.y + 1.0);
		return (1.0 - t)*Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);
	}
}

Vec3 renderer::color(double u, double v, my_rand &rnd)const
{
	Ray r = cam_.get_ray(u, v, rnd);
	return raytrace(r, 0, rnd);
}

void renderer::update(const double *src, double *dest)const
{
	const double INV_WIDTH = 1.0 / (double)WIDTH;
	const double INV_HEIGHT = 1.0 / (double)HEIGHT;
	
	clock_t start = clock();

	#pragma omp parallel
	{
		my_rand rnd(start);
		#pragma omp for
		for (int y = 0; y < HEIGHT; y++) {
			for (int x = 0; x < WIDTH; x++) {
				int index = 3 * (y * WIDTH + x);

				double u = ((double)x + rnd.get()) * INV_WIDTH;
				double v = ((double)y + rnd.get()) * INV_HEIGHT;

				Vec3 color = this->color(u, 1.0 - v, rnd);// 画像的に上下逆だったので、vを反転する
				dest[index + 0] = src[index + 0] + color.x;
				dest[index + 1] = src[index + 1] + color.y;
				dest[index + 2] = src[index + 2] + color.z;
			}
		}
	}
}
