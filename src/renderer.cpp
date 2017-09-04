#include "config.h"

#include <cfloat>
#include <time.h>
#include <omp.h>
#include "hdrloader.h"
#include "renderer.h"

#ifndef max
#define max(a,b) (((a)>(b))?(a):(b))
#endif // max

bool IBL::initialize(int w, int h, const float *p)
{
	w_ = w;
	h_ = h;
	dw_ = w;
	dh_ = h;

	pImage_ = new RenderTarget<Color>(w, h);
	if (!pImage_) return false;

#pragma omp for
	for (int y = 0; y < h; y++) {
		// 上下をひっくり返してコピー
		const float *src = p + (3 * w * (h-1-y));
		Color *dest = pImage_->ref(0, y);
		for (int i = 0; i < w; i++) {
			int idx = 3 * i;
			dest[i].set(src[idx + 0], src[idx + 1], src[idx + 2] );
		}
	}

	return true;
}


inline static int clamp(int src, int v_min, int v_max)
{
	int d = (src < v_min) ? v_min : src;
	return (v_max < d) ? v_max : d;
}

void renderer::edge_detection(const RenderTarget<double> &src, RenderTarget<double> &dest)
{
	int w = src.getWidth();
	int h = src.getHeight();

#pragma omp parallel
	{
#pragma omp for
		for (int y = 0; y < h; y++) {
			int dest_idx = y * w;
			int y0 = (y - 1 < 0) ? 0 : (y - 1);
			int y1 = y;
			int y2 = (h - 1 < y + 1) ? (h - 1) : (y + 1);
			for (int x = 0; x < w; x++) {
				int x0 = (x - 1 < 0) ? 0 : (x - 1);
				int x1 = x;
				int x2 = (w - 1 < x + 1) ? (w - 1) : (x + 1);

				// Sobelフィルタ
				double dx =
					1.0*src.get(y0 * w + x2) - 1.0*src.get(y0 * w + x0) +
					2.0*src.get(y1 * w + x2) - 2.0*src.get(y1 * w + x0) +
					1.0*src.get(y2 * w + x2) - 1.0*src.get(y2 * w + x0);
				double dy =
					1.0*src.get(y2 * w + x0) - 1.0*src.get(y0 * w + x0) +
					2.0*src.get(y2 * w + x1) - 2.0*src.get(y0 * w + x1) +
					1.0*src.get(y2 * w + x2) - 1.0*src.get(y0 * w + x2);

				dx = (dx < 0) ? -dx : dx;
				dy = (dy < 0) ? -dy : dy;

				double v = 0.125 * (dx + dy);
				dest.set(dest_idx, v);
				dest_idx++;
			}
		}
	}
}
void renderer::gauss_blur_x(const RenderTarget<double> &src, RenderTarget<double> &dest)
{
	const double sigma2_inv = -1.0 / (2.0 * 50.0 * 50.0);
	const int KERNEL_SIZE = 100;
	double tbl[KERNEL_SIZE + 1] = { 1.0 };
	double inv_tbl_sum = 1.0;
	for (int i = 1; i <= KERNEL_SIZE; i++) {
		tbl[i] = exp((double)i * (double)i * sigma2_inv);
		inv_tbl_sum += 2.0 * tbl[i];
	}
	inv_tbl_sum = 1.0 / inv_tbl_sum;

	int w = src.getWidth();
	int h = src.getHeight();

#pragma omp parallel
	{
#pragma omp for
		for (int y = 0; y < h; y++) {
			int dest_idx = y * w;
			int src_idx = y * w;
			for (int x = 0; x < w; x++) {

				double s = 0.0;
				for (int i = -KERNEL_SIZE; i <= KERNEL_SIZE; i++) {
					int ix = clamp(x + i, 0, w - 1);
					double weight = tbl[(i < 0) ? (-i) : i];
					s += weight * src.get(src_idx + ix);
				}

				dest.set(dest_idx++, s * inv_tbl_sum);
			}
		}
	}
}

void renderer::gauss_blur_y(const RenderTarget<double> &src, RenderTarget<double> &dest)
{
	const double sigma2_inv = -1.0 / (2.0 * 50.0 * 50.0);
	const int KERNEL_SIZE = 100;
	double tbl[KERNEL_SIZE + 1] = {1.0};
	double inv_tbl_sum = 1.0;
	for (int i = 1; i <= KERNEL_SIZE; i++) {
		tbl[i] = exp((double)i * (double)i * sigma2_inv);
		inv_tbl_sum += 2.0 * tbl[i];
	}
	inv_tbl_sum = 1.0 / inv_tbl_sum;

	int w = src.getWidth();
	int h = src.getHeight();

#pragma omp parallel
	{
#pragma omp for
		for (int y = 0; y < h; y++) {
			int dest_idx = y * w;
			int src_idx = y * w;
			for (int x = 0; x < w; x++) {

				double s = 0.0;
				for (int i = -KERNEL_SIZE; i <= KERNEL_SIZE; i++) {
					int iy = clamp(y + i, 0, h - 1);
					double weight = tbl[(i < 0) ? (-i) : i];
					s += weight * src.get(iy * w + x);
				}

				dest.set(dest_idx++, s * inv_tbl_sum);
				src_idx++;
			}
		}
	}
}

void renderer::compute_normal(const RenderTarget<double> &src, RenderTarget<Vec3> &dest)
{
	int w = src.getWidth();
	int h = src.getHeight();

#pragma omp parallel for
	for (int y = 0; y < h; y++) {
		int dest_idx = src.getIdx(0, y);
		int y0 = (y - 1 < 0) ? 0 : (y - 1);
		int y1 = y;
		int y2 = (h - 1 < y + 1) ? (h - 1) : (y + 1);
		for (int x = 0; x < w; x++) {
			int x0 = (x - 1 < 0) ? 0 : (x - 1);
			int x1 = x;
			int x2 = (w - 1 < x + 1) ? (w - 1) : (x + 1);

			// Sobelフィルタ
			double dx =
				1.0*src.get(x2, y0) - 1.0*src.get(x0, y0) +
				2.0*src.get(x2, y1) - 2.0*src.get(x0, y1) +
				1.0*src.get(x2, y2) - 1.0*src.get(x0, y2);
			double dy =
				1.0*src.get(x0, y2) - 1.0*src.get(x0, y0) +
				2.0*src.get(x1, y2) - 2.0*src.get(x1, y0) +
				1.0*src.get(x2, y2) - 1.0*src.get(x2, y0);

#ifndef SHIPPING
			dest.set(dest_idx++, Vec3(0.25 * dx, 0.25 * dy, sqrt(1.0 - dx * dx + dy * dy)));
#else // SHIPPING
			dest.set(dest_idx++, Vec3(0.25 * dx, 0.25 * dy, src.get(x, y).r));// いったん、zには明るさを入れる
#endif // !SHIPPING
		}
	}
}

void renderer::median_filter(const RenderTarget<Color> &src, RenderTarget<Color> &dest)
{
	int w = src.getWidth();
	int h = src.getHeight();
	const double INV_WIDTH = 1.0 / (double)w;
	const double INV_HEIGHT = 1.0 / (double)h;

#pragma omp parallel
	{
#pragma omp for
		for (int y = 0; y < h; y++) {
			double lum[9];
			double bak[9];
			int id[9];
			int j[9];
			int dest_idx = y * w;
			for (int x = 0; x < w; x++) {

				int idx = 0;
				for (int iy = y-1; iy <= y+1; iy++) {
					int dy = clamp(iy, 0, h - 1);
					for (int ix = x-1; ix <= x+1; ix++) {
						int dx = clamp(ix, 0, w - 1);
						int index = (dy * w + dx);
						lum[idx] = src.get(index).getLuminance();
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

				dest.set(dest_idx++, src.get(index));
			}
		}
	}
}

void renderer::copy(const RenderTarget<Color> &src, RenderTarget<Color> &dest)
{
	int n = dest.getIdxNum();
#pragma omp parallel for
	for (int i = 0; i < n; i++) {
		dest.set(i, src.get(i));
	}
}

void renderer::get_luminance(const RenderTarget<Color> &src, RenderTarget<double> &dest)
{
	int n = dest.getIdxNum();
#pragma omp parallel for
	for (int i = 0; i < n; i++) {
		double l = log(src.get(i).getLuminance() + 1.0);
		dest.set(i, l);
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
	scene_.Append(new Sphere(Vec3(-2, 1.5, -2), 1.0, new GlaredLight(Vec3(0.1, 0.1, 0.1), Color(7.0, 15.0, 2.0), 1.0, 0.4, 0.2)));
}

renderer::~renderer()
{
}

Color renderer::raytrace(Ray r, int depth, my_rand &rnd)const
{
	// Hack(for debugging)
//	return Vec3(0,0,0);
//	return Vec3(rand_.get(), rand_.get(), rand_.get());

	HitRecord rec;
	if (!scene_.hit(r, 0.001, DBL_MAX, rec)) {
		// 交差しなければ、IBLを読み込む
		Vec3 unit_direction = r.direction().normalize();
		return ibl_.get(r.direction().normalize());

		// Hack(IBL がないときの適当な色)
//		double t = 0.5*(unit_direction.y + 1.0);
//		return (1.0 - t)*Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);
	}
	
	Ray scattered;
	Vec3 attenuation; 
	Color emmisive;
	if (50 <= depth) return 0.0;

	bool reflected = rec.mat_ptr->scatter(r, rec, attenuation, emmisive, scattered, rnd);
	double russian_roulette = max(attenuation.x, max(attenuation.y, attenuation.z));

	if (3 <= depth && rnd.get() < russian_roulette) {
		return emmisive;
	}

	return raytrace(scattered, reflected ? (depth + 1) : depth, rnd) * attenuation + emmisive;
}

void renderer::setIBL(int width, int height, const float *image)
{
	ibl_.initialize(width, height, image);
}

void renderer::update(const RenderTarget<Color> *src, RenderTarget<Color> *dest, const RenderTarget<Vec3> *normal_map, int SUPER_SAMPLES) const
{
	const double INV_WIDTH = 1.0 / (double)WIDTH;
	const double INV_HEIGHT = 1.0 / (double)HEIGHT;
	const double INV_SUPER_SAMPLES = 1.0 / (double)SUPER_SAMPLES;

	clock_t start = clock();

	#pragma omp parallel
	{
		my_rand rnd(start);
		#pragma omp for
		for (int y = 0; y < HEIGHT; y++) {
			int index = normal_map->getIdx(0, y);
			for (int x = 0; x < WIDTH; x++) {
				Color color;
				const Vec3 n = normal_map->get(index);

				for (int sy = 0; sy < SUPER_SAMPLES; sy++) {
					for (int sx = 0; sx < SUPER_SAMPLES; sx++) {

//						const double GAZE_SCALE = 0.0;
						const double GAZE_SCALE = 10000.0;

		//				double u = ((double)x + rnd.get() + GAZE_SCALE * n.x) * INV_WIDTH;
		//				double v = ((double)y + rnd.get() + GAZE_SCALE * n.y) * INV_HEIGHT;
						double u = ((double)x + INV_SUPER_SAMPLES*(double)sx + 0.5 * INV_SUPER_SAMPLES + GAZE_SCALE * n.x) * INV_WIDTH;
						double v = ((double)y + INV_SUPER_SAMPLES*(double)sy + 0.5 * INV_SUPER_SAMPLES + GAZE_SCALE * n.y) * INV_HEIGHT;

						Ray r = cam_.get_ray(u, 1.0 - v, rnd);// 画像的に上下逆だったので、vを反転する
		//				Ray r = cam_.get_ray(u, 1.0 - v, rnd, Vec3(-10.0 * n[0], 10.0 * n[1], n[2]));// 画像的に上下逆だったので、vを反転する
						color += raytrace(r, 0, rnd) * INV_SUPER_SAMPLES * INV_SUPER_SAMPLES;
					}
				}
				dest->set(index, src->get(index) + color);
				index++;
			}
		}
	}
}
