namespace
{
    static const float pi = (3.14159265358979323846264338327950288f);

    // SPH kernel implementations
    //
    class Kernel
    {
    private:
        float h, h2, h3, h9;
        float deltaqSq;
        float kPoly, kSpiky;
        explicit Kernel() : h(0.0f), h2(0.0f), h3(0.0f), h9(0.0f), kPoly(0.0), kSpiky(0.0), deltaqSq(0.0f)
        {

        }

    public:
        Kernel(float _h) : h(_h), h2(h* h), h3(h2* h), h9(h3* h3* h3), deltaqSq(0.1f*h*h)
        {
            kPoly = 315.0f / (64.0f * pi * h9);
            kSpiky = 45.0f / (pi * h3 * h3);
        }

        // The poly-6 kernel.
        float poly6(const Eigen::Vector3f& pi, const Eigen::Vector3f& pj) const
        {
            const Eigen::Vector3f pipj = pi - pj;
            const float r = pipj.norm();
            if (r < h)
            {
                const float x = h2 - r*r;
                return kPoly * x * x * x;
            }
            return 0.0f;
        }

        // The "spiky" kernel gradient. Use this for gradient calculations, e.g. when computing lambda 
        //
        Eigen::Vector3f spiky(const Eigen::Vector3f& pi, const Eigen::Vector3f& pj)
        {
            const Eigen::Vector3f pipj = pi - pj;
            const float r = pipj.norm();

            if( r < 1e-6f )
                return Eigen::Vector3f::Zero();
            else if (r < h)
            {
                const float x = (h - r);
                return -((kSpiky * x * x) / r) * pipj;
            }

            return Eigen::Vector3f::Zero();
        }

        // Kernel used for the 
        float scorr()
        {
            const float x = h2 - deltaqSq;
            return kPoly * x * x * x;
        }
    };

}
