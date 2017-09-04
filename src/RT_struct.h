#ifndef RT_STRUCT_H
#define RT_STRUCT_H

#include <cmath>
#include "my_rand.h"

#ifndef PI
#define PI 3.141592653589793826433
#endif// !PI

#ifndef MY_ASSERT
#include <assert.h>
#define MY_ASSERT(x) assert(x)
#endif

#ifndef SAFE_DELETE
#define SAFE_DELETE(p) {if(p)delete(p);(p)=nullptr;}
#endif // SAFE_DELETE

#ifndef SAFE_DELETE_ARRAY
#define SAFE_DELETE_ARRAY(p) {if(p)delete[](p);(p)=nullptr;}
#endif // SAFE_DELETE_ARRAY

struct Vec3 {
	double x, y, z;
	Vec3(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}

	inline Vec3 &operator=(const Vec3 &v) { x = v.x; y = v.y; z = v.z; return *this; }
	inline Vec3 &operator+=(const Vec3 &v) { x += v.x; y += v.y; z += v.z; }
	inline Vec3 &operator-=(const Vec3 &v) { x -= v.x; y -= v.y; z -= v.z; }
	inline Vec3 &operator*=(const Vec3 &v) { x *= v.x; y *= v.y; z *= v.z; }
	inline Vec3 &operator/=(const Vec3 &v) { x /= v.x; y /= v.y; z /= v.z; }

	inline const Vec3 operator+(const Vec3 &b) const {return Vec3(x + b.x, y + b.y, z + b.z);}
	inline const Vec3 operator-(const Vec3 &b) const {return Vec3(x - b.x, y - b.y, z - b.z);}
	inline const Vec3 operator*(const Vec3 &b) const { return Vec3(x * b.x, y * b.y, z * b.z); }
	inline const Vec3 operator*(const double b) const {return Vec3(x * b, y * b, z * b);}
	inline const Vec3 operator/(const Vec3 &b) const { return Vec3(x / b.x, y / b.y, z / b.z); }
	inline const Vec3 operator/(const double b) const {return Vec3(x / b, y / b, z / b);}
	inline const Vec3 operator-() const { return Vec3(-x, -y, -z); }
	inline const double length_sq() const { return x * x + y * y + z * z; }
	inline const double length() const { return sqrt(length_sq()); }
	inline const Vec3 normalize() const { return *this / length(); }
	const Vec3 reflect(const Vec3& n) const;
	bool refract(const Vec3& n, double ni_over_nt, Vec3& refracted) const;

	static Vec3 random_in_unit_disc(my_rand &rnd);
	static Vec3 random_in_unit_sphere(my_rand &rnd);
};
inline const Vec3 operator*(double f, const Vec3 &v) { return v * f; }
inline const Vec3 normalize(const Vec3 &v) {return v * (1.0 / v.length());}
inline const double dot(const Vec3 &v1, const Vec3 &v2) { return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;}
inline const Vec3 cross(const Vec3 &v1, const Vec3 &v2) { return Vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x); }

inline const Vec3 Vec3::reflect(const Vec3& n) const { return *this - n * 2 * dot(*this, n); }
inline bool Vec3::refract(const Vec3& n, double ni_over_nt, Vec3& refracted) const {
	Vec3 uv = this->normalize();
	double dt = dot(uv, n);
	double discriminant = 1.0 - ni_over_nt * ni_over_nt * (1 - dt * dt);
	if (discriminant > 0) {
		refracted = (uv - n * dt)*ni_over_nt - n * sqrt(discriminant);
		return true;
	}
	else
		return false;
}

#ifndef min
#define min(a,b) (((a)<(b))?(a):(b))
#endif // !min

struct ByteColor {
private:
	unsigned char r, g, b;
public:
	ByteColor(unsigned char r0 = 0, unsigned char g0 = 0, unsigned char b0 = 0) { r = r0; g = g0; b = b0; }
};

struct Color {
private:
	double r, g, b;
public:
	Color(double r0 = 0.0, double g0 = 0.0, double b0 = 0.0) { r = r0; g = g0; b = b0; }

	ByteColor getRGB(double scale);
	void set_zero() { r = g = b = 0.0; }
	void set(const Color c) { r = c.r; g = c.g;b = c.b;}
	void set(double r0, double g0, double b0) { r = r0; g = g0; b = b0;}

	inline Color &operator=(const Color &c) { r = c.r; g = c.g; b = c.b; return *this; }
	inline Color &operator+=(const Color &c) { r += c.r; g += c.g; b += c.b; return *this;}
	inline Color &operator-=(const Color &c) { r -= c.r; g -= c.g; b -= c.b; return *this;}
	inline Color &operator*=(const Color &c) { r *= c.r; g *= c.g; b *= c.b; return *this;}
	inline Color &operator/=(const Color &c) { r /= c.r; g /= c.g; b /= c.b; return *this;}
	inline const Color operator+(const Color &c) const { return Color(r + c.r, g + c.g, b + c.b); }
	inline const Color operator-(const Color &c) const { return Color(r - c.r, g - c.g, b - c.b); }
	inline const Color operator*(const Color &c) const { return Color(r * c.r, g * c.g, b * c.b); }
	inline const Color operator/(const Color &c) const { return Color(r / c.r, g / c.g, b / c.b); }
	inline const Color operator*(const Vec3 &v) const { return Color(r * v.x, g * v.y, b * v.z); }
	inline const Color operator+(double v) const { return Color(r + v, g + v, b + v); }
	inline const Color operator-(double v) const { return Color(r - v, g - v, b - v); }
	inline const Color operator*(double v) const { return Color(r * v, g * v, b * v); }

	inline double getLuminance()const { return 0.299 * r + 0.587 * g + 0.114 * b; }
};

template<typename T> inline T Uncharted2Tonemap(const T &x) {
	const double A = 0.15, B = 0.50, C = 0.10, D = 0.20, E = 0.02, F = 0.30;
	return ((((x)*((x)*A+C*B) + D*E) / ((x)*((x)*A+B) + D*F)) - E / F);
}

inline ByteColor Color::getRGB(double scale) {
	// FilmicTonemapping
	// http://filmicworlds.com/blog/filmic-tonemapping-operators/
	Color v = (*this) * 1.0 * scale;  // Hardcoded Exposure Adjustment
//	Color v = (*this) * 16 * scale;  // Hardcoded Exposure Adjustment
	double ExposureBias = 2.0;
	Color curr = Uncharted2Tonemap<Color>(v*ExposureBias);
	const double W = 11.2;
	const double whiteScale = 1.0 / Uncharted2Tonemap<double>(W);
	Color color = curr * whiteScale;
	return ByteColor(
		(unsigned char)((255.99999) * pow(min(color.r, 1.0), 1 / 2.2)),
		(unsigned char)((255.99999) * pow(min(color.g, 1.0), 1 / 2.2)),
		(unsigned char)((255.99999) * pow(min(color.b, 1.0), 1 / 2.2)));
}

template<typename T> class RenderTarget {
	int w_, h_;
	int num_;// w_*h_
	T *a_buf;

public:
	RenderTarget(int width, int height) : w_(width), h_(height) { num_ = width*height; a_buf = new T[num_]; }
	~RenderTarget() { SAFE_DELETE_ARRAY(a_buf); }

	inline int getWidth() const { return w_; }
	inline int getHeight() const { return h_; }
	inline int getIdxNum() const { return num_; }
	inline int getIdx(int x, int y) const { return y * w_ + x; }
	inline T *ref(int idx) { return a_buf + idx; }
	inline T *ref(int x, int y) { return a_buf + getIdx(x, y); }
	inline T get(int idx) const { return a_buf[idx]; }
	inline T get(int x, int y) const { return a_buf[y * w_ + x]; }
	inline void set(int idx, T c) { a_buf[idx] = c; }
	inline const T *ref(int idx) const { return a_buf + idx; }

	static inline const ByteColor getRGB(double &v, double scale) { double d = v * scale; d = (1.0 < d) ? 1.0 : d; unsigned char c = (unsigned char)(255.9999999*d); return ByteColor(c, c, c); }
	static inline const ByteColor getRGB(Vec3 &v, double scale) {
		Vec3 d = v * scale;
		d.x = (1.0 < d.x) ? 1.0 : d.x; d.y = (1.0 < d.y) ? 1.0 : d.y; d.z = (1.0 < d.z) ? 1.0 : d.z;
		d.x = (d.x<-1.0) ? -1.0 : d.x; d.y = (d.y<-1.0) ? -1.0 : d.y; d.z = (d.z<-1.0) ? -1.0 : d.z;
		return ByteColor((unsigned char)(255.99*(d.x*0.5 + 0.5)), (unsigned char)(255.99*(d.y*0.5 + 0.5)), (unsigned char)(255.99*(d.z*0.5 + 0.5)));
	}
	static inline const ByteColor getRGB(Color &v, double scale) { return v.getRGB(scale); }
	static inline const ByteColor getRGB(ByteColor &v, double scale) {
		double r = scale*(double)v.r(); double g = scale*(double)v.g(); double b = scale*(double)v.b();
		r = (r < 255.9) ? r : 255.9; g = (g < 255.9) ? g : 255.9; b = (b < 255.9) ? b : 255.9;
		return v = ByteColor((unsigned char)r, (unsigned char)g, (unsigned char)b);
	}

	static inline void set_zero(double &v) { v = 0; }
	static inline void set_zero(Vec3 &v) { v.x = v.y = v.z = 0.0; }
	static inline void set_zero(Color &v) { v.set_zero(); }
	static inline void set_zero(ByteColor &v) { v = ByteColor(0, 0, 0); }
	void resolve(RenderTarget<ByteColor> *dst, double scale = 1.0)
	{
#pragma omp parallel for
		for (int i = 0; i < num_; i++) {
			//			double tmp = data[i] / (double)steps;// SDR tone mapping
			//			double tmp = 1.0 - exp(-data[i] * coeff);// tone mapping
			//			buf[i] = (unsigned char)(pow(tmp, 1.0 / 2.2) * 255.999);// gamma correct
			dst->set(i, getRGB(a_buf[i], scale));
		}
	}

	void clear()
	{
#pragma omp parallel for
		for (int i = 0; i < num_; i++) {
			set_zero(a_buf[i]);
		}
	}
};

inline Vec3 Vec3::random_in_unit_disc(my_rand &rnd)
{
	do {
		Vec3 p = Vec3(rnd.get(), rnd.get(), 0) * 2.0 - Vec3(1, 1, 0);
		if (dot(p, p) < 1.0) return p;
	} while (true);
}

inline Vec3 Vec3::random_in_unit_sphere(my_rand &rnd)
{
	do {
		Vec3 p = Vec3(rnd.get(), rnd.get(), rnd.get()) * 2.0 - Vec3(1, 1, 1);
		if (dot(p, p) < 1.0) return p;
	} while (true);
}


class Ray
{
private:
	Vec3 o_;
	Vec3 d_;
public:
	Ray() {}
	Ray(const Vec3& o, const Vec3& d) { o_ = o; d_ = d; }
	Vec3 origin() const { return o_; }
	Vec3 direction() const { return d_; }
	Vec3 get(double t) const { return o_ + d_ * t; }
};

class Material;

struct HitRecord
{
	double t;
	Vec3 p;
	Vec3 normal;
	Material *mat_ptr;
};


class Material {
protected:
	static double schlick(double cosine, double ref_idx) {
		double r0 = (1 - ref_idx) / (1 + ref_idx);
		r0 = r0 * r0;
		return r0 + (1 - r0)*pow((1 - cosine), 5);
	}

public:
	virtual bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, my_rand &rnd) const = 0;
};

class Lambertian : public Material {
private:
	Vec3 albedo;
public:
	Lambertian(const Vec3& a) : albedo(a) {}
	virtual bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, my_rand &rnd) const {
		Vec3 target = rec.p + rec.normal + Vec3::random_in_unit_sphere(rnd);
		scattered = Ray(rec.p, target - rec.p);
		attenuation = albedo;
		return true;
	}
};

class Metal : public Material {
private:
	Vec3 albedo;
	float fuzz;
public:
	Metal(const Vec3& a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }
	virtual bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, my_rand &rnd) const {
		Vec3 reflected = r_in.direction().normalize().reflect(rec.normal);
		scattered = Ray(rec.p, reflected + Vec3::random_in_unit_sphere(rnd) * fuzz);
		attenuation = albedo;
		return (dot(scattered.direction(), rec.normal) > 0);
	}
};

class Dielectric : public Material {
public:
	Dielectric(float ri) : ref_idx(ri) {}
	virtual bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, my_rand &rnd) const {
		Vec3 outward_normal;
		Vec3 reflected = r_in.direction().reflect(rec.normal);
		double ni_over_nt;
		attenuation = Vec3(1.0, 1.0, 1.0);
		Vec3 refracted;
		double reflect_prob;
		double cosine;
		if (dot(r_in.direction(), rec.normal) > 0) {
			outward_normal = -rec.normal;
			ni_over_nt = ref_idx;
//          cosine = ref_idx * dot(r_in.direction(), rec.normal) / r_in.direction().length();
			cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
			cosine = sqrt(1 - ref_idx*ref_idx*(1 - cosine*cosine));
		}
		else {
			outward_normal = rec.normal;
			ni_over_nt = 1.0 / ref_idx;
			cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
		}
		if (r_in.direction().refract(outward_normal, ni_over_nt, refracted))
			reflect_prob = schlick(cosine, ref_idx);
		else
			reflect_prob = 1.0;
		if (rnd.get() < reflect_prob)
			scattered = Ray(rec.p, reflected);
		else
			scattered = Ray(rec.p, refracted);
		return true;
	}

	float ref_idx;
};



class Hitable {
public:
	virtual bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const = 0;
	virtual ~Hitable() {}
};

class Sphere : public Hitable {
private:
	Vec3 center_;
	double radius_;
	Material *mat_ptr_;
public:
	Sphere() {}
	Sphere(Vec3 cen, double r, Material *m) : center_(cen), radius_(r), mat_ptr_(m) {};
	~Sphere() { if (mat_ptr_)delete mat_ptr_; mat_ptr_ = nullptr; }
	bool hit(const Ray& r, double tmin, double tmax, HitRecord& rec) const {
		Vec3 oc = r.origin() - center_;
		double a = dot(r.direction(), r.direction());
		double b = dot(oc, r.direction());
		double c = dot(oc, oc) - radius_*radius_;
		double discriminant = b*b - a*c;
		if (0 < discriminant) {
			double temp = (-b - sqrt(discriminant)) / a;
			if (temp < tmax && temp > tmin) {
				rec.t = temp;
				rec.p = r.get(rec.t);
				rec.normal = (rec.p - center_) / radius_;
				rec.mat_ptr = mat_ptr_;
				return true;
			}
			temp = (-b + sqrt(discriminant)) / a;
			if (temp < tmax && temp > tmin) {
				rec.t = temp;
				rec.p = r.get(rec.t);
				rec.normal = (rec.p - center_) / radius_;
				rec.mat_ptr = mat_ptr_;
				return true;
			}
		}
		return false;
	}
};

class HitableList : public Hitable {
private:
	enum{
		LIST_MAX = 255,
	};
	Hitable *list[LIST_MAX];
	int n_;
public:
	HitableList() : n_(0) {}
	~HitableList() {
		for (int i = 0; i < n_; i++) {
			if (list[i]) delete list[i]; list[i] = nullptr;
		}
	}
	void Append(Hitable *l) { if (n_ < LIST_MAX) { list[n_++] = l; } }
	bool hit(const Ray &r, double tmin, double tmax, HitRecord& rec) const
	{
		HitRecord temp_rec;
		bool hit_anything = false;
		double closest_so_far = tmax;
		for (int i = 0; i < n_; i++) {
			if (list[i]->hit(r, tmin, closest_so_far, temp_rec)) {
				hit_anything = true;
				closest_so_far = temp_rec.t;
				rec = temp_rec;
			}
		}
		return hit_anything;
	}
};


#endif // !RT_STRUCT_H
