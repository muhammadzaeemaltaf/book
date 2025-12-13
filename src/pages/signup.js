import React, { useState, useContext, useEffect } from "react";
import { useLocation } from "@docusaurus/router";
import { AuthContext } from "../contexts/AuthContext";
import BackgroundSurvey from "../components/BackgroundSurvey";
import SignupFeedback from "../components/SignupFeedback";
import { getApiUrl } from "../config/api";

const SignupPage = () => {
  const [formData, setFormData] = useState({
    email: "",
    password: "",
    name: "",
  });
  const [backgroundData, setBackgroundData] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState(false);
  const [formStep, setFormStep] = useState(1); // 1 = basic info, 2 = background survey
  const [showPassword, setShowPassword] = useState(false);
  const [passwordStrength, setPasswordStrength] = useState('');
  const authContext = useContext(AuthContext);
  const signup = authContext
    ? authContext.signup
    : async (userData) => {
        // Fallback implementation that calls the API directly
        try {
          // Use helper to compute full API URL for signup
          const apiUrl = getApiUrl("/api/auth/signup");

          const response = await fetch(apiUrl, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify(userData),
          });

          // Check if response is JSON before parsing
          const contentType = response.headers.get("content-type");
          let result;

          if (contentType && contentType.includes("application/json")) {
            result = await response.json();
          } else {
            // If not JSON, try to handle as text or return error
            const text = await response.text();
            if (!response.ok) {
              throw new Error(`HTTP ${response.status}: ${text}`);
            }
            result = { success: response.ok, message: text };
          }

          if (!response.ok) {
            throw new Error(
              result.message || `HTTP ${response.status}: Signup failed`
            );
          }

          // Store the authentication data if returned
          if (result.access_token && result.user) {
            localStorage.setItem("access_token", result.access_token);
            localStorage.setItem("user", JSON.stringify(result.user));
          }

          return { success: true, ...result };
        } catch (error) {
          // Check if this is the "Unexpected token '<'" error
          if (error.message.includes("Unexpected token '<'")) {
            return {
              success: false,
              message:
                "API server is not responding correctly. Please check if the backend server is running.",
            };
          }
          return { success: false, message: error.message };
        }
      };

  useEffect(() => {
    const pwd = formData.password || '';
    const score = (() => {
      let s = 0;
      if (pwd.length >= 8) s += 1;
      if (pwd.length >= 12) s += 1;
      if (/[A-Z]/.test(pwd)) s += 1;
      if (/[0-9]/.test(pwd)) s += 1;
      if (/[^A-Za-z0-9]/.test(pwd)) s += 1;
      return s;
    })();

    if (!pwd) setPasswordStrength('');
    else if (score <= 2) setPasswordStrength('weak');
    else if (score <= 4) setPasswordStrength('moderate');
    else setPasswordStrength('strong');
  }, [formData.password]);

  const handleBasicInfoSubmit = (e) => {
    e.preventDefault();
    setFormStep(2); // Move to background survey
  };

  const handleBackgroundSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const signupData = {
        ...formData,
        ...backgroundData,
      };

      const result = await signup(signupData);
      if (result.success) {
        setSuccess(true);
        setTimeout(() => {
          window.location.href = "/"; // Redirect to home after successful signup
        }, 2000);
      } else {
        setError(result.message || "Signup failed");
      }
    } catch (err) {
      setError(err.message || "An error occurred during signup");
    } finally {
      setLoading(false);
    }
  };

  const updateBackgroundData = (data) => {
    setBackgroundData((prev) => ({ ...prev, ...data }));
  };

  return (
    <div className="container margin-vert--lg auth-page">
      <div className="row" style={{ flex: "1" }}>
        <div className="col col--6 col--offset-3">
          <div className="card auth-card">
            <div className="card__header auth-card__header">
              <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <h1 className="hero__title">Create Your Account</h1>
                <svg
                  width="32"
                  height="32"
                  viewBox="0 0 24 24"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <circle
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="var(--color-primary)"
                    strokeWidth="1.5"
                    fill="rgba(14,165,233,0.06)"
                  />
                  <path
                    d="M9 11.5C9 9.843 10.343 8.5 12 8.5C13.657 8.5 15 9.843 15 11.5C15 13.157 13.657 14.5 12 14.5C10.343 14.5 9 13.157 9 11.5ZM6 17.5C6 15.076 8.243 13.5 12 13.5C15.757 13.5 18 15.076 18 17.5V18H6V17.5Z"
                    stroke="var(--color-primary)"
                    strokeWidth="1.2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </div>
              <p className="hero-subtitle">
                Get started for free — take the first step into adaptive learning.
              </p>
            </div>
            <div className="card__body auth-card__body">
              {error && <SignupFeedback type="error" message={error} />}
              {success && (
                <SignupFeedback
                  type="success"
                  message="Account created successfully! Redirecting..."
                />
              )}

              {!success && (
                <form
                  onSubmit={
                    formStep === 1
                      ? handleBasicInfoSubmit
                      : handleBackgroundSubmit
                  }
                >
                  {formStep === 1 ? (
                    <div>
                      <div className="margin-bottom--md">
                        <label htmlFor="name" className="form-label">
                          Full Name
                        </label>
                        <input
                          type="text"
                          id="name"
                          value={formData.name}
                          onChange={(e) =>
                            setFormData({ ...formData, name: e.target.value })
                          }
                          className="form-control"
                          required
                        />
                      </div>

                      <div className="margin-bottom--md">
                        <label htmlFor="email" className="form-label">
                          Email Address
                        </label>
                        <input
                          type="email"
                          id="email"
                          value={formData.email}
                          onChange={(e) =>
                            setFormData({ ...formData, email: e.target.value })
                          }
                          className="form-control"
                          required
                        />
                      </div>

                      <div className="margin-bottom--lg">
                        <label htmlFor="password" className="form-label">
                          Password
                        </label>
                        <div style={{ position: 'relative' }}>
                          <input
                            type={showPassword ? 'text' : 'password'}
                            id="password"
                            value={formData.password}
                            onChange={(e) =>
                              setFormData({
                                ...formData,
                                password: e.target.value,
                              })
                            }
                            className="form-control"
                            required
                            minLength={8}
                          />
                          <button
                            type="button"
                            onClick={() => setShowPassword(!showPassword)}
                            aria-label={showPassword ? 'Hide password' : 'Show password'}
                            style={{ position: 'absolute', right: 10, top: 10, background: 'transparent', border: 'none', color: 'var(--ifm-font-color-base-inverse)', cursor: 'pointer' }}
                          >
                            {showPassword ? 'Hide' : 'Show'}
                          </button>
                        </div>
                        <div className="form-text text--small">
                          Password must be at least 8 characters long
                        </div>

                        {passwordStrength && (
                          <div className="password-strength mt-2">
                            <div className="password-strength-bar">
                              <div
                                className={`password-strength-fill ${passwordStrength}`}
                                style={{ width: passwordStrength === 'weak' ? '35%' : passwordStrength === 'moderate' ? '65%' : '100%' }}
                              />
                            </div>
                            <div style={{ fontSize: 12, marginLeft: 8, color: 'var(--color-text-muted)' }}>
                              {passwordStrength === 'weak' ? 'Weak' : passwordStrength === 'moderate' ? 'Moderate' : 'Strong'}
                            </div>
                          </div>
                        )}
                      </div>

                      <div className="button-group button-group--block">
                        <div></div> {/* Empty div for spacing */}
                        <button
                          type="submit"
                          className="button button--primary btn-primary"
                        >
                          Continue
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div>
                      <h2 className="text--xl">Technical Background Survey</h2>
                      <p className="margin-bottom--lg">
                        Help us personalize your learning experience by sharing
                        your technical background.
                      </p>

                      <BackgroundSurvey
                        initialData={backgroundData}
                        onDataChange={updateBackgroundData}
                      />

                      <div className="button-group button-group--block margin-top--lg">
                        <button
                          type="button"
                          onClick={() => setFormStep(1)}
                          className="button button--secondary"
                        >
                          Back
                        </button>
                        <button
                          type="submit"
                          disabled={loading}
                          className="button button--primary btn-primary"
                        >
                          {loading ? "Creating Account..." : "Create Account"}
                        </button>
                      </div>
                    </div>
                  )}
                </form>
              )}

              <div className="mt-4 text-center auth-aux-links">
                <p className="text-sm text-gray-600">
                  Already have an account?{" "}
                  <a
                    href="/signin"
                    className="text-blue-600 hover:text-blue-800 font-medium"
                  >
                    Sign in
                  </a>
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SignupPage;
