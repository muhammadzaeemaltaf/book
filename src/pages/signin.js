import React, { useState, useContext } from "react";
import { useLocation } from "@docusaurus/router";
import { AuthContext } from "../contexts/AuthContext";
import { getApiUrl } from "../config/api";
import SignupFeedback from "../components/SignupFeedback";

const SigninPage = () => {
  const [formData, setFormData] = useState({
    email: "",
    password: "",
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const location = useLocation();
  const authContext = useContext(AuthContext);
  const signin = authContext
    ? authContext.signin
    : async (credentials) => {
        // Fallback implementation that calls the API directly
        try {
          // Use helper to compute correct backend URL
          const apiUrl = getApiUrl("/api/auth/signin");

          const response = await fetch(apiUrl, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify(credentials),
          });

          // Check if response is JSON before parsing
          const contentType = response.headers.get("content-type");
          let result;

          if (contentType && contentType.includes("application/json")) {
            result = await response.json();
          } else {
            const text = await response.text();
            if (!response.ok) {
              throw new Error(`HTTP ${response.status}: ${text}`);
            }
            result = { success: response.ok, message: text };
          }

          if (!response.ok) {
            throw new Error(
              result.message || `HTTP ${response.status}: Sign in failed`
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

  const from = location.state?.from?.pathname || "/";

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const result = await signin(formData);
      if (result.success) {
        const returnUrl =
          new URLSearchParams(location.search).get("return") || from;
        window.location.href = returnUrl || "/"; // Redirect to original destination or home
      } else {
        setError(result.message || "Sign in failed");
      }
    } catch (err) {
      setError(err.message || "An error occurred during sign in");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container margin-vert--lg auth-page">
      <div className="row" style={{flex: '1'}}>
        <div className="col col--4 col--offset-4">
          <div className="card auth-card">
            <div className="card__header auth-card__header">
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                }}
              >
                <span className="hero__title">Sign In</span>
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
                    d="M8 12.5L11 15L16 9.5"
                    stroke="var(--color-primary)"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </div>
              <p className="hero-subtitle">Welcome back — sign in to continue.</p>
            </div>
            <div className="card__body auth-card__body">
              {error && <SignupFeedback type="error" message={error} />}

              <form onSubmit={handleSubmit}>
                <div className="mb-4">
                  <label
                    htmlFor="email"
                    className="form-label"
                  >
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

                <div className="mb-6">
                  <label
                    htmlFor="password"
                    className="form-label"
                  >
                    Password
                  </label>
                  <input
                    type="password"
                    id="password"
                    value={formData.password}
                    onChange={(e) =>
                      setFormData({ ...formData, password: e.target.value })
                    }
                    className="form-control"
                    required
                  />
                </div>

                <button
                  type="submit"
                  disabled={loading}
                  className="w-full button button--primary btn-primary"
                >
                  {loading ? "Signing In..." : "Sign In"}
                </button>
              </form>

              <div className="mt-4 text-center auth-aux-links">
                <p className="text-sm text-gray-600">
                  <a href="/forgot-password" className="text-blue-600 hover:text-blue-800">Forgot password?</a>
                </p>
                <p className="text-sm text-gray-600">
                  Don't have an account?{' '}
                  <a
                    href="/signup"
                    className="text-blue-600 hover:text-blue-800 font-medium"
                  >
                    Sign up here
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

export default SigninPage;
