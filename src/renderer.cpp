#include "config.h"

#include <cfloat>
#include <time.h>
#include <omp.h>
#include "hdrloader.h"
#include "renderer.h"


#ifndef MY_ASSERT
#include <assert.h>
#define MY_ASSERT(x) assert(x)
#endif

bool IBL::initialize(int w, int h, const float *p)
{
	w_ = w;
	h_ = h;
	dw_ = w;
	dh_ = h;

	pImage_ = new double[3 * w * h];
	if (!pImage_) return false;

#pragma omp for
	for (int y = 0; y < h; y++) {
		// 上下をひっくり返してコピー
		const float *src = p + (3 * w * (h-1-y));
		double *dest = pImage_ + 3 * w * y;
		for (int i = 0; i < 3 * w; i++) {
			dest[i] = src[i];
		}
	}

	return true;
}


inline static double RGB2Y(double r, double g, double b)
{
	return 0.299 * r + 0.587 * g + 0.114 * b;
}

inline static int clamp(int src, int v_min, int v_max)
{
	int d = (src < v_min) ? v_min : src;
	return (v_max < d) ? v_max : d;
}

void renderer::edge_detection(const double *src, double *dest)const
{
#pragma omp parallel
	{
#pragma omp for
		for (int y = 0; y < HEIGHT; y++) {
			int dest_idx = 3 * y * WIDTH;
			int y0 = (y - 1 < 0) ? 0 : (y - 1);
			int y1 = y;
			int y2 = (HEIGHT - 1 < y + 1) ? (HEIGHT - 1) : (y + 1);
			for (int x = 0; x < WIDTH; x++) {
				int x0 = (x - 1 < 0) ? 0 : (x - 1);
				int x1 = x;
				int x2 = (WIDTH - 1 < x + 1) ? (WIDTH - 1) : (x + 1);

				// Sobelフィルタ
				double dx =
					1.0*src[3 * (y0 * WIDTH + x2)] - 1.0*src[3 * (y0 * WIDTH + x0)] +
					2.0*src[3 * (y1 * WIDTH + x2)] - 2.0*src[3 * (y1 * WIDTH + x0)] +
					1.0*src[3 * (y2 * WIDTH + x2)] - 1.0*src[3 * (y2 * WIDTH + x0)];
				double dy =
					1.0*src[3 * (y2 * WIDTH + x0)] - 1.0*src[3 * (y0 * WIDTH + x0)] +
					2.0*src[3 * (y2 * WIDTH + x1)] - 2.0*src[3 * (y0 * WIDTH + x1)] +
					1.0*src[3 * (y2 * WIDTH + x2)] - 1.0*src[3 * (y0 * WIDTH + x2)];

				dx = (dx < 0) ? -dx : dx;
				dy = (dy < 0) ? -dy : dy;

#ifndef SHIPPING
				dest[dest_idx + 1] = dest[dest_idx + 2] =
#endif // !SHIPPING
					dest[dest_idx] = 0.125 * (dx + dy);
				dest_idx += 3;
			}
		}
	}
}
void renderer::gauss_blur_x(const double *src, double *dest) const
{
	const double sigma2_inv = -1.0 / (2.0 * 20.0 * 20.0);
	const int KERNEL_SIZE = 100;
	double tbl[KERNEL_SIZE + 1] = { 1.0 };
	double tbl_sum = 0.0;
	for (int i = 1; i <= KERNEL_SIZE; i++) {
		tbl[i] = exp((double)i * (double)i * sigma2_inv);
		tbl_sum += 2.0 * tbl[i];
	}

#pragma omp parallel
	{
#pragma omp for
		for (int y = 0; y < HEIGHT; y++) {
			int dest_idx = 3 * y * WIDTH;
			int src_idx = 3 * y * WIDTH;
			for (int x = 0; x < WIDTH; x++) {

				double s = 0.0;
				for (int i = -KERNEL_SIZE; i <= KERNEL_SIZE; i++) {
					int ix = clamp(x + i, 0, WIDTH - 1);
					double w = tbl[(i < 0) ? (-i) : i];
					s += w * src[src_idx + 3 * ix];
				}

#ifndef SHIPPING
					dest[dest_idx + 1] = dest[dest_idx + 2] = 
#endif // !SHIPPING
					dest[dest_idx] = s / tbl_sum;
				dest_idx += 3;
			}
		}
	}
}

void renderer::gauss_blur_y(const double *src, double *dest) const
{
	const double sigma2_inv = -1.0 / (2.0 * 20.0 * 20.0);
	const int KERNEL_SIZE = 100;
	double tbl[KERNEL_SIZE + 1] = {1.0};
	double tbl_sum = 0.0;
	for (int i = 1; i <= KERNEL_SIZE; i++) {
		tbl[i] = exp((double)i * (double)i * sigma2_inv);
		tbl_sum += 2.0 * tbl[i];
	}

#pragma omp parallel
	{
#pragma omp for
		for (int y = 0; y < HEIGHT; y++) {
			int dest_idx = 3 * y * WIDTH;
			int src_idx = 3 * y * WIDTH;
			for (int x = 0; x < WIDTH; x++) {
				int src_idx = 3 * x;

				double s = 0.0;
				for (int i = -KERNEL_SIZE; i <= KERNEL_SIZE; i++) {
					int iy = clamp(y + i, 0, HEIGHT - 1);
					double w = tbl[(i < 0) ? (-i) : i];
					s += w * src[src_idx + 3 * iy * WIDTH];
				}

#ifndef SHIPPING
				dest[dest_idx + 1] = dest[dest_idx + 2] =
#endif // !SHIPPING
					dest[dest_idx] = s / tbl_sum;
				dest_idx += 3;
				src_idx += 3;
			}
		}
	}
}

void renderer::compute_normal(const double *src, double *dest)const
{
#pragma omp parallel
	{
#pragma omp for
		for (int y = 0; y < HEIGHT; y++) {
			int dest_idx = 3 * y * WIDTH;
			int y0 = (y - 1 < 0) ? 0 : (y - 1);
			int y1 = y;
			int y2 = (HEIGHT - 1 < y + 1) ? (HEIGHT - 1) : (y + 1);
			for (int x = 0; x < WIDTH; x++) {
				int x0 = (x - 1 < 0) ? 0 : (x - 1);
				int x1 = x;
				int x2 = (WIDTH - 1 < x + 1) ? (WIDTH - 1) : (x + 1);

				// Sobelフィルタ
				double dx =
					1.0*src[3 * (y0 * WIDTH + x2)] - 1.0*src[3 * (y0 * WIDTH + x0)] +
					2.0*src[3 * (y1 * WIDTH + x2)] - 2.0*src[3 * (y1 * WIDTH + x0)] +
					1.0*src[3 * (y2 * WIDTH + x2)] - 1.0*src[3 * (y2 * WIDTH + x0)];
				double dy =
					1.0*src[3 * (y2 * WIDTH + x0)] - 1.0*src[3 * (y0 * WIDTH + x0)] +
					2.0*src[3 * (y2 * WIDTH + x1)] - 2.0*src[3 * (y0 * WIDTH + x1)] +
					1.0*src[3 * (y2 * WIDTH + x2)] - 1.0*src[3 * (y0 * WIDTH + x2)];

				dest[dest_idx + 0] = 0.25 * dx;
				dest[dest_idx + 1] = 0.25 * dy;
#ifndef SHIPPING
				dest[dest_idx + 2] = sqrt(1.0 - dx * dx + dy * dy);
#endif // !SHIPPING
				dest[dest_idx + 2] = src[3 * (y * WIDTH + x)];// いったん、zには明るさを入れる
				dest_idx += 3;
			}
		}
	}
}

void renderer::median_filter(const double *src, double *dest)const
{
	const double INV_WIDTH = 1.0 / (double)WIDTH;
	const double INV_HEIGHT = 1.0 / (double)HEIGHT;

#pragma omp parallel
	{
#pragma omp for
		for (int y = 0; y < HEIGHT; y++) {
			double lum[9];
			double bak[9];
			int id[9];
			int j[9];
			int dest_idx = 3 * y * WIDTH;
			for (int x = 0; x < WIDTH; x++) {

				int idx = 0;
				for (int iy = y-1; iy <= y+1; iy++) {
					int dy = clamp(iy, 0, HEIGHT - 1);
					for (int ix = x-1; ix <= x+1; ix++) {
						int dx = clamp(ix, 0, WIDTH - 1);
						int index = 3 * (dy * WIDTH + dx);
						lum[idx] = RGB2Y(src[index + 0], src[index + 1], src[index + 2]);
						bak[idx] = lum[idx];
						id[idx] = index;
						j[idx] = index;
						idx++;
					}
				}
				MY_ASSERT(idx == 9);
				// FORGETFULL SELECTION
				// 最初の6個の内、最大の値と最小の値は無視する
				// 次の１つを足した残り5つに関して、最大の値と最小の値は無視する
				// 次の１つを足した残り4つに関して、最大の値と最小の値は無視する
				// 次の１つを足した残り3つに関して、中間の値がメディアン
				unsigned int flag = 0;
				for(int loop = 5; loop < 9; loop++){
					int min_idx = -1; double l_min = 100000000.0;
					int max_idx = -1; double l_max = -1.0;
					for (int i = 0; i <= loop; i++) {
						if (flag & (1 << i))continue;
						if (lum[i] < l_min) { min_idx = i; l_min = lum[i]; }
						if (l_max <= lum[i]) { max_idx = i; l_max = lum[i]; }
					}
					flag |= (1 << min_idx) | (1 << max_idx);
				}

				int index = -1;
				int n;
				for (n = 0; n < 9; n++) {
					if (!(flag & (1 << n))) { index = id[n]; break; }
				}

				int count0 = 0;
				int count1 = 0;
				for (int i = 0; i < 9; i++) {
					if (bak[i] < bak[n]) { count0++; }
					if (bak[n] < bak[i]) { count1++; }
				}
				MY_ASSERT(count0 <= 4 && count1 <= 4);


				dest[dest_idx + 0] = src[index + 0];
				dest[dest_idx + 1] = src[index + 1];
				dest[dest_idx + 2] = src[index + 2];
				dest_idx += 3;
			}
		}
	}
}

void renderer::copy(const double *src, double *dest)const
{
#pragma omp parallel for
	for (int i = 0; i < 3 * WIDTH * HEIGHT; i++) {
		dest[i] = src[i];
	}
}

void renderer::get_luminance(const double *src, double *dest) const
{
#pragma omp parallel for
	for (int i = 0; i < WIDTH * HEIGHT; i++) {
		int idx = 3 * i;
#ifndef SHIPPING
		dest[idx + 1] = dest[idx + 2] =
#endif // !SHIPPING
			dest[idx] = log(RGB2Y(src[idx], src[idx + 1], src[idx + 2])+1.0);
	}
}

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
	Vec3 from(-13, 3, -3);
	Vec3 lookat(0, 0.5, 0);
	Vec3 up(0, 1, 0);
	double fov = 20.0;
	double aspect = (double)WIDTH / (double)HEIGHT;
	double dist_to_focus = 10.0;
	double aperture = .1;

	cam_.init(from, lookat, up, fov, aspect, aperture, dist_to_focus);

	// scene
	double R = cos(PI / 4);
	Material *p = new Lambertian(Vec3(0.5, 0.5, 0.5));
	scene_.Append(new Sphere(Vec3(0, -1000, 0), 1000, p));
	scene_.Append(new Sphere(Vec3(0, 1, 0), 1.0, new Dielectric(1.5)));
	scene_.Append(new Sphere(Vec3(+4, 1, 0), 1.0, new Lambertian(Vec3(0.4, 0.2, 0.1))));
	scene_.Append(new Sphere(Vec3(-4, 1, 0), 1.0, new Metal(Vec3(0.7, 0.6, 0.5), 0.0)));
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
//		double t = 0.5*(unit_direction.y + 1.0);
//		return (1.0 - t)*Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);
		return ibl_.get(r.direction().normalize());
	}
}

void renderer::setIBL(int width, int height, const float *image)
{
	ibl_.initialize(width, height, image);
}

void renderer::update(const double *src, double *dest, const double *normal_map)const
{
	const double INV_WIDTH = 1.0 / (double)WIDTH;
	const double INV_HEIGHT = 1.0 / (double)HEIGHT;
	
	clock_t start = clock();

	#pragma omp parallel
	{
		my_rand rnd(start);
		#pragma omp for
		for (int y = 0; y < HEIGHT; y++) {
			int index = 3 * y * WIDTH;
			for (int x = 0; x < WIDTH; x++) {
				const double *n = normal_map + index;

				const double GAZE_SCALE = 700.0;
//				const double GAZE_SCALE = 7000.0;

				double u = ((double)x + rnd.get() + GAZE_SCALE * n[0]) * INV_WIDTH;
				double v = ((double)y + rnd.get() + GAZE_SCALE * n[1]) * INV_HEIGHT;

				Ray r = cam_.get_ray(u, 1.0 - v, rnd);// 画像的に上下逆だったので、vを反転する
//				Ray r = cam_.get_ray(u, 1.0 - v, rnd, Vec3(-10.0 * n[0], 10.0 * n[1], n[2]));// 画像的に上下逆だったので、vを反転する
				Vec3 color = raytrace(r, 0, rnd);

				dest[index + 0] = src[index + 0] + color.x;
				dest[index + 1] = src[index + 1] + color.y;
				dest[index + 2] = src[index + 2] + color.z;

				index += 3;
			}
		}
	}
}
