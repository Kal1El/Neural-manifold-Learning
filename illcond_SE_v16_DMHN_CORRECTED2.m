%==========================================================================
%  POWER SYSTEM STATE ESTIMATION  –  v19 (WLS from v19 + Stabilized DMHN from v16)
%  Three methods on three test systems
%
%  METHOD 1 – WLS-Std  : Pure Newton with CORRECT sign (x = x - G\g)
%  METHOD 2 – WLS-Mod  : Levenberg-Marquardt with CORRECT sign
%  METHOD 3 – RNN-DMHN : v16 STABILIZED DMHN (leakage + trust region + Armijo)
%==========================================================================

function illcond_SE_v19()

    clc; close all; rng(42);

    measMode    = 'HYBRID';
    methodNames = {'WLS-Std', 'WLS-Mod (LM)', 'RNN-DMHN'};
    colors      = {[0 0 0], [0 0.45 0.74], [0.85 0.33 0.10]};
    nMethods    = 3;
    sysNums     = [11, 13, 20];
    sysNames    = {'11-bus (Well-cond.)', '13-bus (Ill-cond.)', '20-bus (Moderate)'};
    nSys        = 3;

    tol_grad = 1e-4;
    tol_dx   = 1e-4;
    rmse_lim = 1e-1;

    SEP  = repmat('=', 1, 96);
    SEP2 = repmat('-', 1, 96);
    fprintf('\n%s\n  POWER SYSTEM STATE ESTIMATION  –  v19\n%s\n\n', SEP, SEP);

    all_hists = cell(nSys, nMethods);
    all_rmse  = zeros(nSys, nMethods);
    all_conv  = false(nSys, nMethods);
    chi2_tgt  = zeros(nSys, 1);
    sys_data  = cell(nSys, 1);
    pf_valid  = false(nSys, 1);

    %======================================================================
    %  BUILD SYSTEMS
    %======================================================================
    fprintf('  BUILDING TEST SYSTEMS\n%s\n', SEP2);

    for si = 1:nSys
        num  = sysNums(si);
        mpc  = build_mpc(num);
        n    = mpc.n;
        nb_  = size(mpc.branch, 1);
        pb   = get_pmu_buses(num, n);

        [Ybus, Yf] = build_Ybus(mpc, n);

        fprintf('  Running power flow for %s ...\n', sysNames{si});
        [th_true, Vm_true, pf_ok, pf_meth] = run_pf(mpc, Ybus, n);

        if pf_ok
            fprintf('    Converged  [%s]\n', pf_meth);
            pf_valid(si) = true;
        else
            fprintf('    PF FAILED – using flat start\n');
        end

        fprintf('    Vm: ['); fprintf(' %.4f', Vm_true(1:min(n,5)));
        if n>5, fprintf(' ...'); end; fprintf(' ] p.u.\n');
        fprintf('    Th: ['); fprintf(' %.2f', th_true(1:min(n,5))*180/pi);
        if n>5, fprintf(' ...'); end; fprintf(' ] deg\n');

        [z_true, meas] = build_meas(mpc, Ybus, Yf, th_true, Vm_true, ...
                                    n, nb_, pb, measMode);
        rng(42);
        z   = z_true + meas.sigma .* randn(size(z_true));
        Rd  = 1 ./ meas.sigma.^2;
        nm  = numel(z);
        Nst = 2*n - 1;
        dof = max(nm - Nst, 1);
        chi2_tgt(si) = 0.5*dof + 4.0*sqrt(0.5*dof);

        % Flat start
        x0  = [zeros(n-1,1); ones(n,1)];

        H0      = jac_H(x0, meas, Ybus, Yf, mpc, n, nm);
        HWH0    = H0' * bsxfun(@times, H0, Rd);
        kappaHWH = cond(full(HWH0));

        fprintf('    n=%d  nm=%d  dof=%d  kappa(G0)=%.2e  chi2_tgt=%.1f\n', ...
            n, nm, dof, kappaHWH, chi2_tgt(si));

        d.mpc=mpc; d.n=n; d.nb=nb_; d.Ybus=Ybus; d.Yf=Yf;
        d.th_true=th_true; d.Vm_true=Vm_true;
        d.z=z; d.meas=meas; d.Rd=Rd; d.nm=nm; d.x0=x0;
        d.kappaHWH=kappaHWH; d.dof=dof;
        d.pf_ok=pf_ok; d.pf_meth=pf_meth;
        sys_data{si} = d;
        fprintf('\n');
    end

    %======================================================================
    %  STATE ESTIMATION
    %======================================================================
    fprintf('%s\n  STATE ESTIMATION FROM FLAT START\n%s\n', SEP, SEP2);

    for si = 1:nSys
        d    = sys_data{si};
        n    = d.n;
        meas = d.meas; z=d.z; Rd=d.Rd; nm=d.nm; x0=d.x0;
        Vm_t = d.Vm_true;
        Ybus = d.Ybus; Yf=d.Yf; mpc=d.mpc;
        Jchi = chi2_tgt(si);
        pf_ok_si = d.pf_ok;

        fprintf('\n  [%s]  true state via: %s\n', sysNames{si}, d.pf_meth);
        fprintf('  %-18s %6s %6s %10s %12s %10s %8s  [status]\n', ...
            'Method','Conv','Iter','RMSE_V','J(x)','||g||_s','Time(s)');

        % ---- WLS Standard (v19 CORRECTED sign) ----
        p1 = struct('MAX_ITER', 150, 'tol', tol_grad, 'tol_dx', tol_dx);
        t0 = tic;
        [x1, h1] = se_wls_std(z, meas, Ybus, Yf, mpc, n, nm, Rd, x0, p1);
        t1 = toc(t0);
        rv1 = rmse_V(x1, Vm_t, n);
        h1  = audit(h1, x1, z, meas, Ybus, Yf, mpc, n, nm, Rd, Jchi, rv1, ...
                    tol_grad, tol_dx, rmse_lim, pf_ok_si, 'WLS-Std');

        % ---- WLS Modified (v19 CORRECTED sign) ----
        p2 = struct('MAX_ITER', 300, 'tol', tol_grad, 'tol_dx', tol_dx);
        t0 = tic;
        [x2, h2] = se_wls_lm(z, meas, Ybus, Yf, mpc, n, nm, Rd, x0, p2);
        t2 = toc(t0);
        rv2 = rmse_V(x2, Vm_t, n);
        h2  = audit(h2, x2, z, meas, Ybus, Yf, mpc, n, nm, Rd, Jchi, rv2, ...
                    tol_grad, tol_dx, rmse_lim, pf_ok_si, 'WLS-Mod');

        % ---- RNN-DMHN (v16 STABILIZED version with alpha=0.3) ----
        p3 = build_dmhn_params(d.kappaHWH, n, Jchi, tol_grad, tol_dx);
        t0 = tic;
        [x3, h3] = se_rnn_dmhn_v16_stabilized(z, meas, Ybus, Yf, mpc, n, nm, Rd, x0, p3);
        t3 = toc(t0);
        rv3 = rmse_V(x3, Vm_t, n);
        h3  = audit(h3, x3, z, meas, Ybus, Yf, mpc, n, nm, Rd, Jchi, rv3, ...
                    tol_grad, tol_dx, rmse_lim, pf_ok_si, 'RNN-DMHN');

        hs = {h1, h2, h3};
        rs = [rv1, rv2, rv3];
        ts = [t1, t2, t3];

        for mi = 1:nMethods
            all_hists{si,mi} = hs{mi};
            all_rmse(si,mi)  = rs(mi);
            all_conv(si,mi)  = hs{mi}.conv;
            csym = '✗'; if hs{mi}.conv, csym='✓'; end
            fprintf('  %-18s %6s %6d %10.3e %12.2f %10.3e %8.3f  [%s]\n', ...
                methodNames{mi}, csym, hs{mi}.iters, rs(mi), ...
                hs{mi}.J_final, hs{mi}.gn_final, ts(mi), hs{mi}.status);
        end
    end

    %======================================================================
    %  FIGURES
    %======================================================================
    fprintf('\n%s\n  GENERATING FIGURES\n%s\n', SEP, SEP2);

    fnt      = 'Times New Roman';
    fs_title = 16; fs_label = 14; fs_tick = 12; fs_ann = 11;
    lw       = 2.0;

    fig_specs = { ...
        'Fig_Energy',   'Objective  J(x)',          'J'; ...
        'Fig_Gradient', 'Scaled gradient ||g||_s',  'gn'; ...
        'Fig_Step',     'Step size  ||dx||_\infty', 'step'};

    for fi = 1:3
        fname  = fig_specs{fi,1};
        ytitle = fig_specs{fi,2};
        fld    = fig_specs{fi,3};

        figure('Color','w','Name',fname,'Position',[30+fi*30, 30, 1300, 900]);

        for si = 1:nSys
            for mi = 1:nMethods
                ax = subplot(nSys, nMethods, (si-1)*nMethods + mi);
                hold(ax,'on');

                h_  = all_hists{si,mi};
                dat = abs(h_.(fld)(:));
                dat(~isfinite(dat) | dat <= 0) = NaN;

                if any(isfinite(dat))
                    semilogy(ax, 1:numel(dat), dat, '-', ...
                        'Color', colors{mi}, 'LineWidth', lw);
                end

                if fi == 1
                    yline(ax, chi2_tgt(si), '--g', 'LineWidth', 1.5, 'Alpha', 0.7);
                end

                if h_.conv
                    ki = min(h_.iters, numel(dat));
                    v  = dat(ki);
                    if isfinite(v) && v > 0
                        semilogy(ax, ki, v, 'k*', 'MarkerSize',14,'LineWidth',2);
                    end
                end

                rv = all_rmse(si,mi);
                if     rv > 0.05, rc=[0.8 0 0];
                elseif rv > 0.01, rc=[0.8 0.6 0];
                else,             rc=[0 0.5 0];
                end
                text(ax, 0.03, 0.97, sprintf('RMSE=%.2e', rv), ...
                    'Units','normalized','VerticalAlignment','top', ...
                    'FontSize',fs_ann,'Color',rc,'FontWeight','bold');

                if h_.conv
                    text(ax,0.97,0.03,'✓ CONV','Units','normalized', ...
                        'HorizontalAlignment','right','VerticalAlignment','bottom', ...
                        'FontSize',fs_ann,'Color',[0 0.5 0],'FontWeight','bold');
                else
                    text(ax,0.97,0.03,'✗ FAIL','Units','normalized', ...
                        'HorizontalAlignment','right','VerticalAlignment','bottom', ...
                        'FontSize',fs_ann,'Color',[0.8 0 0],'FontWeight','bold');
                end

                set(ax,'YScale','log','FontName',fnt,'FontSize',fs_tick, ...
                    'FontWeight','bold','LineWidth',1.2,'Box','on');
                grid(ax,'on');

                if si==1
                    title(ax, methodNames{mi},'FontName',fnt,'FontSize',fs_title, ...
                        'Color',colors{mi},'FontWeight','bold');
                end
                if mi==1
                    ylabel(ax, ytitle,'FontName',fnt,'FontSize',fs_label,'FontWeight','bold');
                    text(ax,-0.22,0.5, sysNames{si},'Units','normalized', ...
                        'Rotation',90,'HorizontalAlignment','center', ...
                        'FontName',fnt,'FontSize',fs_ann,'FontWeight','bold');
                end
                if si==nSys
                    xlabel(ax,'Iteration','FontName',fnt,'FontSize',fs_label,'FontWeight','bold');
                end
            end
        end

        sgtitle(sprintf('SE Convergence — %s  (tol=1e-4)', ytitle), ...
            'FontName',fnt,'FontSize',fs_title+2,'FontWeight','bold');
        print('-dpng','-r300', fname);
        fprintf('  Saved: %s.png\n', fname);
    end

    %======================================================================
    %  SUMMARY TABLE
    %======================================================================
    fprintf('\n%s\n  FINAL SUMMARY\n%s\n', SEP, SEP2);
    fprintf('┌──────────────────────┬──────────────────┬────────────┬────────────┬────────────┐\n');
    fprintf('│ System               │ Method           │ Converged  │ RMSE_V     │ Iterations │\n');
    fprintf('├──────────────────────┼──────────────────┼────────────┼────────────┼────────────┤\n');
    for si = 1:nSys
        for mi = 1:nMethods
            cs = '✗ NO '; if all_conv(si,mi), cs='✓ YES'; end
            fprintf('│ %-20s │ %-16s │ %-10s │ %10.3e │ %10d │\n', ...
                sysNames{si}, methodNames{mi}, cs, ...
                all_rmse(si,mi), all_hists{si,mi}.iters);
        end
        if si < nSys
            fprintf('├──────────────────────┼──────────────────┼────────────┼────────────┼────────────┤\n');
        end
    end
    fprintf('└──────────────────────┴──────────────────┴────────────┴────────────┴────────────┘\n');

    fprintf('\n%s\n  DONE  –  3 PNG figures saved\n%s\n', SEP, SEP);
end

%==========================================================================
%  METHOD 1 : WLS STANDARD (v19 CORRECTED SIGN)
%  x = x - G\g  (Newton descent)
%==========================================================================
function [x, hist] = se_wls_std(z, meas, Ybus, Yf, mpc, n, nm, Rd, x0, par)
    x    = clip_x(x0, n);
    hist = init_hist(par.MAX_ITER);

    for k = 1:par.MAX_ITER
        [~, ~, J, g, G] = compute_all(x, z, meas, Ybus, Yf, mpc, n, nm, Rd);

        gn        = sgi(g, G);
        hist.J(k) = J;
        hist.gn(k)= gn;

        rcG = rcond(full(G));
        if ~isfinite(rcG) || rcG < 1e-15
            G = G + 1e-6 * max(abs(diag(G))) * eye(size(G));
        end
        dx = G \ g;
        dx(~isfinite(dx)) = 0;

        hist.step(k) = norm(dx, inf);
        x = clip_x(x - dx, n);      % CORRECT SIGN: subtract

        if gn <= par.tol && hist.step(k) <= par.tol_dx
            hist = trim(hist, k, 'CONVERGED');
            hist.conv = true;
            return;
        end
    end
    hist = trim(hist, par.MAX_ITER, 'MAX_ITER');
end

%==========================================================================
%  METHOD 2 : WLS MODIFIED (v19 CORRECTED SIGN + Adaptive Lambda)
%  x = x - (G + lambda*diag(G))\g
%==========================================================================
function [x, hist] = se_wls_lm(z, meas, Ybus, Yf, mpc, n, nm, Rd, x0, par)
    x      = clip_x(x0, n);
    hist   = init_hist(par.MAX_ITER);
    lambda = 1e-3;
    lam_dn = 0.1;
    lam_up = 10.0;

    for k = 1:par.MAX_ITER
        [~, ~, J, g, G] = compute_all(x, z, meas, Ybus, Yf, mpc, n, nm, Rd);

        gn        = sgi(g, G);
        hist.J(k) = J;
        hist.gn(k)= gn;

        dg   = max(abs(diag(G)), 1e-12 * (max(abs(diag(G))) + eps));
        A    = G + lambda * diag(dg);
        dx   = A \ g;
        dx(~isfinite(dx)) = 0;

        hist.step(k) = norm(dx, inf);
        x_new = clip_x(x - dx, n);     % CORRECT SIGN: subtract

        [~, ~, J_new, ~, ~] = compute_all(x_new, z, meas, Ybus, Yf, mpc, n, nm, Rd);
        pred = g' * dx - 0.5 * (dx' * G * dx);
        act  = J - J_new;
        if isfinite(pred) && pred > 0
            rho = act / pred;
        else
            rho = -1;
        end

        if rho > 0.25
            x = x_new;
            lambda = max(lambda * lam_dn, 1e-12);
        else
            lambda = min(lambda * lam_up, 1e10);
        end

        if gn <= par.tol && hist.step(k) <= par.tol_dx
            hist = trim(hist, k, 'CONVERGED');
            hist.conv = true;
            return;
        end
    end
    hist = trim(hist, par.MAX_ITER, 'MAX_ITER');
end

%==========================================================================
%  METHOD 3 : RNN-DMHN (v16 STABILIZED VERSION with alpha=0.3)
%  
%  This is the stabilized DMHN from v16 with:
%    - Weight decay via norm clipping (prevents Hebbian runaway)
%    - Trust region via clip_step (limits step size)
%    - Armijo condition via rho > eta_rej (accepts only good steps)
%    - Adaptive mu based on condition number
%    - SVD fallback for ill-conditioned systems
%    - Damped step (alpha=0.3 instead of 1.0)
%==========================================================================
function [x, hist] = se_rnn_dmhn_v16_stabilized(z, meas, Ybus, Yf, mpc, n, nm, Rd, x0, par)
    Nst = 2*n - 1;
    x = clip_x(x0, n);
    K = par.MAX_ITER;
    hist = init_hist(K);
    tol_g = par.tol_grad;
    tol_dx = par.tol_dx;
    x_best = x;
    J_best = inf;
    mu = par.mu0;
    nu = 2.0;
    sigma_mean = mean(1 ./ sqrt(max(Rd, 1e-12)));
    ds = dmhn_init(Nst, par);
    
    for k = 1:K
        x = clip_x(x, n);
        [r, H, J_t, g, G] = compute_all(x, z, meas, Ybus, Yf, mpc, n, nm, Rd);
        gn_v = sgi(g, G);
        hist.J(k) = J_t;
        hist.gn(k) = gn_v;
        hist.mu(k) = mu;
        
        if J_t < J_best
            J_best = J_t;
            x_best = x;
        end
        
        if J_t <= par.chi2 && (gn_v <= tol_g || (k > 1 && hist.step(k-1) <= tol_dx))
            hist = trim(hist, k, 'TRUE_CONV');
            hist.conv = true;
            x = x_best;
            return;
        end
        
        [Q_eff, g_eff, T_leak, ds] = dmhn_effective_system(G, g, r, sigma_mean, x, n, ds, par);
        
        diagQ = max(abs(diag(Q_eff)), 1e-14);
        Q_lm = Q_eff + mu * diag(diagQ);
        Q_lm = 0.5 * (Q_lm + Q_lm');
        
        [Qp, Ds] = struct_precond(Q_lm);
        dp = solve_svd(Qp, Ds .* g_eff, par.svd_tol);
        d = Ds .* dp;
        d(~isfinite(d)) = 0;
        d = clip_step(d, n, par);
        
        gTd = g_eff' * d;
        if ~isfinite(gTd) || gTd >= 0
            d = -g_eff;
            d = clip_step(d, n, par);
            gTd = g_eff' * d;
            if gTd >= 0
                d = zeros(Nst, 1);
            end
        end
        
        alpha = 0.3;  % Damped step (was 1.0 - too aggressive for ill-conditioned)
        accepted = false;
        rho_k = -inf;
        pred_full = -g_eff' * d - 0.5 * (d' * Q_lm * d);
        if ~isfinite(pred_full) || pred_full <= 0
            pred_full = 1e-4 * max(-gTd, 1e-12);
        end
        
        for ls = 1:par.max_ls
            xt = clip_x(x + alpha * d, n);
            [~, ~, Jt, ~, ~] = compute_all(xt, z, meas, Ybus, Yf, mpc, n, nm, Rd);
            act = J_t - Jt;
            rho_k = act / max(alpha * pred_full, 1e-18);
            if isfinite(Jt) && act > 0 && rho_k > par.eta_rej
                accepted = true;
                break;
            end
            alpha = alpha * 0.5;
            if alpha < 1e-14
                break;
            end
        end
        
        if accepted
            hist.step(k) = norm(alpha * d, inf);
            x = clip_x(x + alpha * d, n);
            mu = mu * max(1/3, 1 - (2*rho_k - 1)^3);
            nu = 2.0;
        else
            hist.step(k) = 0;
            mu = min(par.mu_max, mu * nu);
            nu = 2 * nu;
        end
    end
    x = x_best;
    hist = trim(hist, K, 'NOT_CONVERGED');
    [~, ~, Jf, gf, Gf] = compute_all(x, z, meas, Ybus, Yf, mpc, n, nm, Rd);
    if Jf <= par.chi2 && sgi(gf, Gf) <= tol_g
        hist.conv = true;
        hist.status = 'TRUE_CONV';
    end
end

%==========================================================================
%  DMHN CORE FUNCTIONS (from v16)
%==========================================================================
function [Q_eff, g_eff, T_leak, dmhn_state] = ...
        dmhn_effective_system(G, g, r, sigma_mean, x, n, dmhn_state, par)

    u_c = g / max(norm(g), 1e-8);
    T_leak = sqrt(max(abs(diag(G)), 1e-14));
    phi_uc = tanh(u_c / par.phi_scale);

    A = dmhn_state.A;
    Au = A * phi_uc;
    W_D = (Au * Au') * par.wd_scale;

    Wi = dmhn_state.Wi;
    I_D = Wi * phi_uc;

    Q_eff = G + W_D + diag(T_leak);
    Q_eff = 0.5 * (Q_eff + Q_eff');
    g_eff = g - I_D;

    g_dir = g / max(norm(g), 1e-8);
    dmhn_state.A = A + par.eta_A * (phi_uc * g_dir');
    dmhn_state.Wi = Wi + par.eta_Wi * (phi_uc * phi_uc');
    
    nA = norm(dmhn_state.A, 'fro');
    if nA > par.A_max
        dmhn_state.A = dmhn_state.A * (par.A_max / nA);
    end
    nW = norm(dmhn_state.Wi, 'fro');
    if nW > par.Wi_max
        dmhn_state.Wi = dmhn_state.Wi * (par.Wi_max / nW);
    end
end

function ds = dmhn_init(Nst, par)
    ds.A = eye(Nst) * par.A_init;
    ds.Wi = eye(Nst) * par.Wi_init;
end

function p = build_dmhn_params(kHWH, n, chi2, tol_g, tol_dx)
    p.MAX_ITER = 500;
    p.chi2 = chi2;
    p.tol_grad = tol_g;
    p.tol_dx = tol_dx;
    p.mu0 = min(1.0, 1e-3 * max(1, log10(kHWH)));
    p.mu_max = 1e10;
    p.eta_rej = 1e-4;
    p.max_ls = 25;
    p.svd_tol = 1e-10;
    p.max_angle_step = 0.08;
    p.max_volt_step = 0.04;
    p.phi_scale = 1.0;
    p.wd_scale = 0.1;
    p.eta_A = 0.01;
    p.eta_Wi = 0.005;
    p.A_init = 0.01;
    p.Wi_init = 0.001;
    p.A_max = 10.0;
    p.Wi_max = 5.0;
    if kHWH > 1e5
        p.wd_scale = 0.5;
        p.eta_A = 0.05;
    end
end

%==========================================================================
%  CORE COMPUTATION (from v16)
%==========================================================================
function [r, H, J, g, G] = compute_all(x, z, meas, Ybus, Yf, mpc, n, nm, Rd)
    x = clip_x(x, n);
    hx = h_meas(x, meas, Ybus, Yf, mpc, n);
    r = hx - z;
    r(~isfinite(r)) = 0;
    H = jac_H(x, meas, Ybus, Yf, mpc, n, nm);
    H(~isfinite(H)) = 0;
    G = H' * bsxfun(@times, H, Rd);
    G = 0.5 * (G + G');
    g = H' * (Rd .* r);
    J = 0.5 * (r' * (Rd .* r));
end

%==========================================================================
%  MEASUREMENT MODEL h(x) (from v16)
%==========================================================================
function hx = h_meas(x, meas, Ybus, Yf, mpc, n)
    theta = [0; x(1:n-1)];
    Vmag = max(0.01, x(n:end));
    V = Vmag .* exp(1i * theta);
    Sinj = V .* conj(Ybus * V);
    If = Yf * V;
    fr = max(1, min(n, round(mpc.branch(:, 1))));
    Sf = V(fr) .* conj(If);
    hx = [real(Sinj); imag(Sinj); Vmag; real(Sf); imag(Sf)];
    if strcmpi(meas.mode, 'HYBRID')
        pb = meas.pmu_buses(:);
        hx = [hx; Vmag(pb) .* cos(theta(pb)); Vmag(pb) .* sin(theta(pb))];
    end
    hx(~isfinite(hx)) = 0;
end

%==========================================================================
%  JACOBIAN H = dh/dx (from v16)
%==========================================================================
function H = jac_H(x, meas, Ybus, Yf, mpc, n, nm)
    Nst = numel(x);
    theta = [0; x(1:n-1)];
    Vmag = max(0.01, x(n:end));
    V = Vmag .* exp(1i * theta);
    nb_ = size(mpc.branch, 1);
    fr = max(1, min(n, round(mpc.branch(:, 1))));
    H = zeros(nm, Nst);
    Ibus = Ybus * V;
    Vnorm = V ./ max(abs(V), 1e-9);
    dSth = 1i * diag(V) * conj(diag(Ibus) - Ybus * diag(V));
    dSVm = diag(V) * conj(Ybus * diag(Vnorm)) + diag(Ibus) * conj(diag(Vnorm));
    H(1:n, 1:n-1) = real(dSth(:, 2:n));
    H(1:n, n:2*n-1) = real(dSVm);
    H(n+1:2*n, 1:n-1) = imag(dSth(:, 2:n));
    H(n+1:2*n, n:2*n-1) = imag(dSVm);
    H(2*n+1:3*n, n:2*n-1) = eye(n);
    If = Yf * V;
    Vf = V(fr);
    Sf = Vf .* conj(If);
    Vfn = Vf ./ max(abs(Vf), 1e-9);
    Cf = sparse((1:nb_)', fr, 1, nb_, n);
    dSf_Va = 1j * (diag(Sf) * Cf - diag(Vf) * conj(Yf * diag(V)));
    dSf_Vm = diag(conj(If) .* Vfn) * Cf + diag(Vf) * conj(Yf * diag(Vnorm));
    rP_start = 3*n + 1;
    rP_end = min(3*n + nb_, nm);
    rQ_start = 3*n + nb_ + 1;
    rQ_end = min(3*n + 2*nb_, nm);
    nb_P = rP_end - rP_start + 1;
    nb_Q = rQ_end - rQ_start + 1;
    if nb_P > 0
        H(rP_start:rP_end, 1:n-1) = real(full(dSf_Va(1:nb_P, 2:n)));
        H(rP_start:rP_end, n:2*n-1) = real(full(dSf_Vm(1:nb_P, :)));
    end
    if nb_Q > 0
        H(rQ_start:rQ_end, 1:n-1) = imag(full(dSf_Va(1:nb_Q, 2:n)));
        H(rQ_start:rQ_end, n:2*n-1) = imag(full(dSf_Vm(1:nb_Q, :)));
    end
    if strcmpi(meas.mode, 'HYBRID')
        pb = meas.pmu_buses(:);
        np = numel(pb);
        for k = 1:np
            bi = pb(k);
            thi = bi - 1;
            Vc = n + bi - 1;
            rr = 3*n + 2*nb_ + k;
            ri = 3*n + 2*nb_ + np + k;
            if rr > nm || ri > nm
                continue;
            end
            if thi >= 1 && thi <= n-1
                H(rr, thi) = -Vmag(bi) * sin(theta(bi));
                H(ri, thi) = Vmag(bi) * cos(theta(bi));
            end
            H(rr, Vc) = cos(theta(bi));
            H(ri, Vc) = sin(theta(bi));
        end
    end
    H(~isfinite(H)) = 0;
end

%==========================================================================
%  MEASUREMENT BUILDER (from v16)
%==========================================================================
function [z, meas] = build_meas(mpc, Ybus, Yf, theta, Vmag, n, nb_, pb, mode)
    V = Vmag .* exp(1i * theta);
    Sinj = V .* conj(Ybus * V);
    If = Yf * V;
    fr = max(1, min(n, round(mpc.branch(:, 1))));
    Sf = V(fr) .* conj(If);
    zs = [real(Sinj); imag(Sinj); Vmag; real(Sf); imag(Sf)];
    sig_s = [0.02*ones(n,1); 0.025*ones(n,1); 0.006*ones(n,1);
            0.02*ones(nb_,1); 0.025*ones(nb_,1)];
    if strcmpi(mode, 'HYBRID')
        np = numel(pb);
        zp = [Vmag(pb).*cos(theta(pb)); Vmag(pb).*sin(theta(pb))];
        z = [zs; zp];
        sigma = [sig_s; 0.001*ones(2*np,1)];
    else
        z = zs;
        sigma = sig_s;
        pb = [];
    end
    meas.mode = upper(mode);
    meas.n = n;
    meas.nb = nb_;
    meas.pmu_buses = pb;
    meas.sigma = sigma;
end

%==========================================================================
%  POWER FLOW (from v16)
%==========================================================================
function [theta, Vmag, conv, method_used] = run_pf(mpc, Ybus, n)
    theta = zeros(n,1);
    Vmag = ones(n,1);
    conv = false;
    method_used = 'NONE';

    if exist('runpf','file') == 2 || exist('runpf','file') == 6
        try
            mpc_mp = mpc_to_matpower(mpc, n);
            mpopt = mpoption('verbose',0,'out.all',0,'pf.tol',1e-8,'pf.max_it',300);
            r = runpf(mpc_mp, mpopt);
            if r.success
                theta = r.bus(:,9) * pi/180;
                Vmag = r.bus(:,8);
                conv = true;
                method_used = 'MATPOWER-NR';
                return;
            end
            mpopt2 = mpoption(mpopt,'pf.alg','FDBX','pf.max_it',500);
            r = runpf(mpc_mp, mpopt2);
            if r.success
                theta = r.bus(:,9) * pi/180;
                Vmag = r.bus(:,8);
                conv = true;
                method_used = 'MATPOWER-FDBX';
                return;
            end
        catch
        end
    end

    bt = mpc.bus(:,2);
    Psch = mpc.bus(:,5) - mpc.bus(:,3);
    Qsch = mpc.bus(:,6) - mpc.bus(:,4);
    sl = find(bt == 1);
    pv = find(bt == 2);
    pq = find(bt == 3);
    thf = setdiff((1:n)', sl);
    vf = pq;

    starts = { ...
        struct('th', zeros(n,1), 'Vm', ones(n,1),            'name','FlatStart'), ...
        struct('th', 0.1*randn(n,1), 'Vm', 1+0.05*randn(n,1), 'name','PertStart')};

    for sp = 1:numel(starts)
        th0 = starts{sp}.th; Vm0 = starts{sp}.Vm;
        [th_t, Vm_t, ok] = nr_pf(th0, Vm0, Psch, Qsch, Ybus, n, sl, pv, pq, thf, vf, mpc, 500, 0.1);
        if ok
            theta = th_t;
            Vmag = Vm_t;
            conv = true;
            method_used = sprintf('InternalNR(%s)', starts{sp}.name);
            return;
        end
    end
    method_used = 'FAILED-FlatStart';
end

function [theta, Vmag, conv] = nr_pf(th0, Vm0, Psch, Qsch, Ybus, n, sl, pv, pq, thf, vf, mpc, maxiter, step_lim)
    theta = th0; Vmag = Vm0; conv = false;
    for it = 1:maxiter
        V = Vmag .* exp(1i*theta);
        Sinj = V .* conj(Ybus*V);
        mis = [Psch(thf) - real(Sinj(thf));
               Qsch(vf)  - imag(Sinj(vf))];
        if norm(mis,inf) < 1e-6, conv=true; return; end
        J = pf_jac(Vmag, theta, Ybus, n, thf, vf);
        if isempty(J) || rcond(J) < 1e-14 || any(~isfinite(J(:))), return; end
        dx = J \ mis;
        dx(~isfinite(dx)) = 0;
        if norm(dx,inf) > step_lim
            dx = dx * (step_lim / norm(dx,inf));
        end
        nt = numel(thf);
        theta(thf) = theta(thf) + dx(1:nt);
        Vmag(vf) = Vmag(vf) + dx(nt+1:end);
        Vmag = max(0.5, min(1.5, Vmag));
        Vmag(pv) = mpc.bus(pv,7);
    end
end

function J = pf_jac(Vmag, theta, Ybus, n, thi, vi)
    V = Vmag .* exp(1i*theta);
    I = Ybus*V;
    dV = diag(V);
    dVn = diag(V ./ max(abs(V),1e-9));
    dSth = 1i*dV*conj(diag(I)-Ybus*dV);
    dSV = dV*conj(Ybus*dVn) + diag(I)*conj(dVn);
    J = [real(dSth(thi,thi)), real(dSV(thi,vi));
         imag(dSth(vi,thi)),  imag(dSV(vi,vi))];
    J(~isfinite(J)) = 0;
end

%==========================================================================
%  MATPOWER BUS-TYPE MAPPING (from v16)
%==========================================================================
function mpc_mp = mpc_to_matpower(mpc, n)
    mpc_mp.version = '2';
    mpc_mp.baseMVA = 100;

    mpc_mp.bus = zeros(n,13);
    mpc_mp.bus(:,1) = (1:n)';

    bt_int = mpc.bus(:,2);
    bt_mp = ones(n,1);
    bt_mp(bt_int == 2) = 2;
    bt_mp(bt_int == 1) = 3;
    mpc_mp.bus(:,2) = bt_mp;

    mpc_mp.bus(:,3) = mpc.bus(:,3)*100;
    mpc_mp.bus(:,4) = mpc.bus(:,4)*100;
    mpc_mp.bus(:,5) = 0;
    mpc_mp.bus(:,6) = 0;
    mpc_mp.bus(:,7) = 1;
    mpc_mp.bus(:,8) = mpc.bus(:,7);
    mpc_mp.bus(:,9) = 0;
    mpc_mp.bus(:,10) = 100;
    mpc_mp.bus(:,11) = 1;
    mpc_mp.bus(:,12) = 1.5;
    mpc_mp.bus(:,13) = 0.5;

    gen_idx = find(bt_int == 1 | bt_int == 2);
    if isempty(gen_idx), gen_idx = 1; end
    ng = numel(gen_idx);
    mpc_mp.gen = zeros(ng,21);
    for i = 1:ng
        b = gen_idx(i);
        mpc_mp.gen(i,1) = b;
        mpc_mp.gen(i,2) = mpc.bus(b,5)*100;
        mpc_mp.gen(i,3) = mpc.bus(b,6)*100;
        mpc_mp.gen(i,4) = 9999;
        mpc_mp.gen(i,5) = -9999;
        mpc_mp.gen(i,6) = mpc.bus(b,7);
        mpc_mp.gen(i,7) = 100;
        mpc_mp.gen(i,8) = 1;
        mpc_mp.gen(i,9) = 9999;
        mpc_mp.gen(i,10)= -9999;
    end

    nb_ = size(mpc.branch,1);
    mpc_mp.branch = zeros(nb_,13);
    mpc_mp.branch(:,1:5) = mpc.branch(:,1:5);
    mpc_mp.branch(:,6:8) = 9999;
    tap = mpc.branch(:,6);
    tap(tap < 0.5 | ~isfinite(tap)) = 1;
    mpc_mp.branch(:,9) = tap;
    mpc_mp.branch(:,10) = 0;
    mpc_mp.branch(:,11) = 1;
    mpc_mp.branch(:,12) = -360;
    mpc_mp.branch(:,13) = 360;
end

%==========================================================================
%  AUDIT (from v16)
%==========================================================================
function hist = audit(hist, x, z, meas, Ybus, Yf, mpc, n, nm, Rd, J_chi, rmseV, tol_g, tol_dx, rmse_lim, pf_ok, name)
    [~, ~, Jf, gf, Gf] = compute_all(x, z, meas, Ybus, Yf, mpc, n, nm, Rd);
    gn_f = sgi(gf, Gf);
    d = max(abs(diag(Gf)), 1e-14);
    dxf = norm(gf ./ d, inf);
    hist.J_final = Jf;
    hist.gn_final = gn_f;
    hist.dx_final = dxf;
    hist.rmseV = rmseV;

    J_ok = isfinite(Jf) && Jf <= J_chi;
    g_ok = isfinite(gn_f) && gn_f <= tol_g;
    dx_ok = isfinite(dxf) && dxf <= tol_dx;
    last_st = NaN;
    if isfield(hist,'step') && ~isempty(hist.step)
        v = hist.step(isfinite(hist.step));
        if ~isempty(v), last_st = v(end); end
    end
    step_ok = isfinite(last_st) && last_st <= tol_dx;
    rmse_fail = pf_ok && (~isfinite(rmseV) || rmseV > rmse_lim);

    if rmse_fail
        hist.conv = false;
        hist.status = 'FALSE_CONV_HIGH_RMSE';
    elseif ~isfinite(Jf) || ~isfinite(gn_f)
        hist.conv = false;
        hist.status = 'DIVERGED';
    elseif J_ok && (g_ok || dx_ok || step_ok)
        hist.conv = true;
        hist.status = 'TRUE_CONV';
    elseif J_ok
        hist.conv = false;
        hist.status = 'J_OK_GRAD_STALLED';
    elseif g_ok && dx_ok
        hist.conv = false;
        hist.status = 'GRAD_OK_HIGH_J';
    else
        hist.conv = false;
        hist.status = 'NOT_CONVERGED';
    end
    hist.audit_method = name;
end

%==========================================================================
%  UTILITIES (from v16)
%==========================================================================
function val = sgi(g, G)
    g(~isfinite(g)) = 0;
    if isempty(G) || size(G,1) ~= numel(g)
        val = norm(g,inf); return;
    end
    d = abs(diag(G));
    d(~isfinite(d)) = 0;
    pos = d(d>0);
    sf = 1e-12 * (max(pos) + eps);
    d = max(d, sf);
    gs = g ./ sqrt(d);
    gs(~isfinite(gs)) = 0;
    val = norm(gs,inf);
end

function rv = rmse_V(x, Vtrue, n)
    if isempty(x) || any(~isfinite(x))
        rv = inf; return;
    end
    rv = sqrt(mean((x(n:end) - Vtrue(:)).^2));
end

function x = clip_x(x, n)
    x(~isfinite(x)) = 0;
    x(1:n-1) = max(-pi/2+1e-8, min(pi/2-1e-8, x(1:n-1)));
    x(n:end) = max(0.5+1e-8,   min(1.5-1e-8,  x(n:end)));
end

function d = clip_step(d, n, par)
    d(~isfinite(d)) = 0;
    na = norm(d(1:min(n-1,end)),inf);
    if na > par.max_angle_step && na > 0
        d(1:n-1) = d(1:n-1) * (par.max_angle_step / na);
    end
    if numel(d) >= 2*n-1
        nv = norm(d(n:end),inf);
        if nv > par.max_volt_step && nv > 0
            d(n:end) = d(n:end) * (par.max_volt_step / nv);
        end
    end
end

function [Qp, Ds] = struct_precond(Q)
    d = abs(diag(Q));
    d = max(d, 1e-12*(max(d(d>0))+eps));
    Ds = 1 ./ sqrt(d);
    Ds(~isfinite(Ds)) = 1;
    Qp = bsxfun(@times, bsxfun(@times, Q, Ds), Ds');
    Qp = 0.5*(Qp + Qp');
end

function d = solve_svd(Q, g, tol)
    Q(~isfinite(Q)) = 0; g(~isfinite(g)) = 0;
    Q = 0.5*(Q+Q');
    rc = rcond(Q);
    if ~isfinite(rc), rc=0; end
    if rc < tol
        [U,S,V] = svd(Q,'econ');
        sv = diag(S);
        if isempty(sv) || max(sv)<=0, d=-g; return; end
        dmp = 1e-6*max(sv);
        d = -V*((sv./(sv.^2+dmp^2)).*(U'*g));
    else
        try
            [L,f] = chol(Q,'lower');
            if f==0, d = -(L'\(L\g));
            else,    d = -(Q\g); end
        catch
            d = -(Q\g);
        end
    end
    if any(~isfinite(d)), d=-g; end
end

function hist = init_hist(K)
    hist.J = nan(K,1);
    hist.gn = nan(K,1);
    hist.step = nan(K,1);
    hist.mu = nan(K,1);
    hist.iters = K;
    hist.conv = false;
    hist.status = 'MAX_ITER';
end

function hist = trim(hist, k, status)
    Kf = max(2, k);
    hist.J = hist.J(1:Kf);
    hist.gn = hist.gn(1:Kf);
    hist.step = hist.step(1:Kf);
    hist.mu = hist.mu(1:Kf);
    hist.iters = k;
    hist.status = status;
end

%==========================================================================
%  SYSTEM DATA (exact Hadi Sadat / Quah Kah Meng convention)
%==========================================================================
function mpc = build_mpc(num)
    switch num
        case 11
            busdat = [ ...
                1   1   1.024   0   0      0      0       0       0   0; ...
                2   3   1.0     0   0      0      0       0       0   0; ...
                3   3   1.0     0   0      0     -0.128  -0.062   0   0; ...
                4   3   1.0     0   0      0      0       0       0   0; ...
                5   3   1.0     0   0      0     -0.165  -0.080   0   0; ...
                6   3   1.0     0   0      0     -0.090  -0.068   0   0; ...
                7   3   1.0     0   0      0      0       0       0   0; ...
                8   3   1.0     0   0      0      0       0       0   0; ...
                9   3   1.0     0   0      0     -0.026  -0.009   0   0; ...
               10   3   1.0     0   0      0      0       0       0   0; ...
               11   3   1.0     0   0      0     -0.158  -0.057   0   0];
            linedat = [ ...
                1  2  0        0.0706  0       1; ...
                2  3  0        0.154   0       1; ...
                2  4  0.0377   0.0413  0       1; ...
                3  5  0.1228   0.1803  0       1; ...
                4  5  0        0.4593  0       1; ...
                4  6  0        0.0176  0       1; ...
                4  7  0.6114   0.8117  0       1; ...
                7  8  0.6209   0.2167  0       1; ...
                8  9  0.0718   0.7179  0       1; ...
                8 10  0.4097   0.5600  0       1; ...
               10 11  0.0264   0.2646  0       1];

        case 13
            busdat = [ ...
                1   1   1.0     0   0    0   1.65   0.56   0   0; ...
                2   3   1.05    0   0    0   0      0      0   0; ...
                3   3   1.05    0   0    0   0      0      0   0; ...
                4   3   1.05    0   0    0   0      0      0   0; ...
                5   2   1.0     0   0    0   0      0      0   0; ...
                6   2   1.037   0   0.5  0   0.05   0.03   0   0; ...
                7   3   1.05    0   0    0   0      0      0   0; ...
                8   2   1.1     0   0    0   0      0      0   0; ...
                9   2   0.943   0   0.5  0   0      0      0   0; ...
               10   2   1.1     0   0    0   0      0      0   0; ...
               11   3   1.05    0   0    0   0.05   0.03   0   0; ...
               12   3   1.05    0   0    0   0.05   0.032  0   0; ...
               13   3   1.05    0   0    0   0      0      0   0];
            linedat = [ ...
                1  2  0.004   0.085   0.05   1; ...
                1  3  0.004   0.0947  0.10   1; ...
                5  4  0.004   0.0947  0.10   1; ...
                4  3  0.0074  0.143   0.218  1; ...
                6  2  0.0481  0.459   0.123  1; ...
                6  7  0.009   0.108   0.008  1; ...
                8  3  0.0121  0.233   0.356  1; ...
                7  8  0       0.15    0      1; ...
                9 10  0.0105  0.202   0.31   1; ...
               10 11  0       0.15    0      1; ...
               11 12  0.0086  0.1665  0.254  1; ...
               12 13  0.0075  0.1465  0.224  1; ...
               13  8  0       0.15    0      1];

        case 20
            busdat_mw = [ ...
                1   3   1   0   0    0   150   30   0   0; ...
                2   3   1   0   0    0   10    0    0   0; ...
                3   3   1   0   0    0   0     0    0   0; ...
                4   3   1   0   0    0   380   60   0   0; ...
                5   3   1   0   0    0   0     0    0   0; ...
                6   3   1   0   0    0   0     0    0   0; ...
                7   3   1   0   0    0   20    0    0   0; ...
                8   3   1   0   0    0   10    20   0   0; ...
                9   3   1   0   0    0   0     0    0   0; ...
               10   3   1   0   0    0   50    10   0   0; ...
               11   3   1   0   0    0   0     0    0   0; ...
               12   3   1   0   0    0   0     0    0   0; ...
               13   3   1   0   0    0   0     0    0   0; ...
               14   3   1   0   0    0   0     10   0   0; ...
               15   3   1   0   0    0   0     0    0   0; ...
               16   3   1   0   0    0   10    0    0   0; ...
               17   2   1   0   100  0   0     0    0   0; ...
               18   2   1   0   100  0   0     0    0   0; ...
               19   2   1   0   100  0   0     0    0   0; ...
               20   1   1   0   0    0   0     0    0   0];
            busdat = busdat_mw;
            busdat(:,5:8) = busdat_mw(:,5:8) / 100;
            linedat_pct = [ ...
                1 20  0.5   5      0.012    1; ...
                2  8  0.5   5      0.0335   1; ...
                2 16  0     5      0        1; ...
                2 17  60    60     0        1; ...
                3  5  20    20     0        1; ...
                3 20  0.11  1.52   0.04285  1; ...
                4 17  3     4      0.025    1; ...
                4 20  5     10     0.125    1; ...
                5 14  30    40     0        1; ...
                6 15  5     10     0        1; ...
                6 16  60    80     0        1; ...
                6 17  0.6   8      0.02     1; ...
                7 12  0.5   5      0.025    1; ...
                9 10  0.5   5      0.025    1; ...
                9 19  0.1   3      0.05     1; ...
               10 11  0     30     0        1; ...
               11 12  2     40     0.012    1; ...
               11 13  0     15     0        1; ...
               13 18  0.5   6      0.015    1; ...
               14 19  0.1   1      0        1; ...
               15 18  0.15  1.5    0        1; ...
               15 20  2     4      0.0335   1];
            linedat = linedat_pct;
            linedat(:,3) = linedat_pct(:,3) / 100;
            linedat(:,4) = linedat_pct(:,4) / 100;

        otherwise
            error('Unknown system number %d', num);
    end

    n   = size(busdat, 1);
    nb_ = size(linedat, 1);

    mpc.bus = zeros(n, 7);
    for i = 1:n
        mpc.bus(i,:) = [busdat(i,1), busdat(i,2), busdat(i,7), busdat(i,8), ...
                        busdat(i,5), busdat(i,6), busdat(i,3)];
    end

    mpc.branch = zeros(nb_, 6);
    for k = 1:nb_
        tap = linedat(k,6);
        if tap == 0, tap = 1; end
        mpc.branch(k,:) = [linedat(k,1:5), tap];
    end
    mpc.n = n;
end

function pb = get_pmu_buses(num, n)
    switch num
        case 11, pb = [1 4 8];
        case 13, pb = [1 6 8];
        case 20, pb = [1 17 20];
        otherwise, pb = [1 round(n/2) n];
    end
    pb = pb(pb>=1 & pb<=n);
end

%==========================================================================
%  ADMITTANCE MATRIX (from v16)
%==========================================================================
function [Ybus, Yf] = build_Ybus(mpc, n)
    nb_ = size(mpc.branch,1);
    Ybus = sparse(n,n);
    Yf = zeros(nb_,n);
    for k = 1:nb_
        f_ = max(1,min(n, round(mpc.branch(k,1))));
        t_ = max(1,min(n, round(mpc.branch(k,2))));
        r_ = mpc.branch(k,3);
        x_ = mpc.branch(k,4);
        if abs(x_) < 1e-3, x_ = sign(x_+eps)*1e-3; end
        bsh = mpc.branch(k,5);
        tap = mpc.branch(k,6); if tap==0, tap=1; end
        den = r_^2 + x_^2;
        ys = 0; if den >= 1e-12, ys=(r_-1i*x_)/den; end
        yff = (ys + 1i*bsh/2)/tap^2;
        yft = -ys/conj(tap);
        ytf = -ys/tap;
        ytt = ys + 1i*bsh/2;
        Ybus(f_,f_) = Ybus(f_,f_) + yff;
        Ybus(f_,t_) = Ybus(f_,t_) + yft;
        Ybus(t_,f_) = Ybus(t_,f_) + ytf;
        Ybus(t_,t_) = Ybus(t_,t_) + ytt;
        Yf(k,f_) = yff; Yf(k,t_) = yft;
    end
    Yf = sparse(Yf);
end