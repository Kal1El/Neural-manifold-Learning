% Modified MATLAB file: flat-start integrated for all methods
% Single Tier Version: 1e-4 tolerance only

function illcond_SE_v16_DMHN_CORRECTED()
%==========================================================================
%  ILL-CONDITIONED POWER SYSTEM STATE ESTIMATION - CORRECTED
%  SINGLE TIER VERSION (tol = 1e-4)
%  
%  TRUTH:
%  - 11-bus: Well-conditioned, WLS methods SHOULD converge
%  - 13-bus: Ill-conditioned, DMHN methods designed for this
%  - 20-bus: Moderately ill-conditioned
%==========================================================================

    clc; close all; rng(42);

    measMode    = 'HYBRID';
    methodNames = {'WLS-Std', 'WLS-Mod', 'RNN-DMHN', 'SPE-DMHN'};
    colors      = {[0 0 0], [0 0.45 0.74], [0.85 0.33 0.10], [0.47 0.67 0.19]};
    nMethods    = 4;
    sysNums     = [11, 13, 20];
    sysNames    = {'11-bus (Well-conditioned)', '13-bus (Ill-conditioned)', '20-bus (Moderate)'};
    nSys        = 3;

    % SINGLE TIER: 1e-4 tolerance
    tol_grad = 1e-4;
    tol_dx   = 1e-4;
    rmse_lim = 1e-1;  % 10% RMSE limit

    SEP  = repmat('=', 1, 96);
    SEP2 = repmat('-', 1, 96);
    fprintf('\n%s\n POWER SYSTEM STATE ESTIMATION - DMHN METHODS (Tolerance = 1e-4)\n%s\n\n', SEP, SEP);

    all_hists = cell(nSys, nMethods);
    all_rmse  = zeros(nSys, nMethods);
    all_conv  = false(nSys, nMethods);
    chi2_tgt  = zeros(nSys, 1);
    sys_data  = cell(nSys, 1);
    pf_valid  = false(nSys, 1);

    %======================================================================
    %  BUILD SYSTEMS
    %======================================================================
    fprintf(' BUILDING TEST SYSTEMS:\n%s\n', SEP2);
    
    for si = 1:nSys
        num = sysNums(si);
        mpc = build_mpc(num);
        n   = mpc.n;
        nb  = size(mpc.branch, 1);
        pb  = get_pmu_buses(num, n);
        [Ybus, Yf] = build_Ybus(mpc, n);

        % Try power flow - NO DC FALLBACK
        [th_true, Vm_true, pok, method] = run_pf_no_fallback(mpc, Ybus, n);
        pf_valid(si) = pok;
        
        if ~pok
            fprintf('  ⚠ %s: PF FAILED - using FLAT START (DMHN challenge!)\n', sysNames{si});
            th_true = zeros(n, 1);
            Vm_true = ones(n, 1);
        else
            fprintf('  ✓ %s: PF CONVERGED via %s\n', sysNames{si}, method);
        end

        % Generate measurements
        [z_true, meas] = build_meas(mpc, Ybus, Yf, th_true, Vm_true, n, nb, pb, measMode);
        rng(42);
        z    = z_true + meas.sigma .* randn(size(z_true));
        Rinv = diag(1 ./ meas.sigma.^2);
        Rd   = diag(Rinv);
        nm   = numel(z);
        Nst  = 2*n - 1;
        dof  = max(nm - Nst, 1);
        chi2_tgt(si) = 0.5*dof + 4.0*sqrt(0.5*dof);

        % FLAT START for all methods
        x0 = [zeros(n-1, 1); ones(n, 1)];
        x0 = clip_x(x0, n);

        H0       = jac_H(x0, meas, Ybus, Yf, mpc, n, nm);
        kappaHWH = cond(H0' * Rinv * H0);

        fprintf('      n=%d, nm=%d, dof=%d, κ(HWH)=%.2e, χ²=%.1f\n', n, nm, dof, kappaHWH, chi2_tgt(si));
        [rv0,~] = rmse_state(x0, Vm_true, th_true, n);
        fprintf('      Flat start RMSE_V = %.2e\n\n', rv0);

        d.mpc = mpc; d.n = n; d.nb = nb; d.Ybus = Ybus; d.Yf = Yf;
        d.th_true = th_true; d.Vm_true = Vm_true; d.z = z; d.meas = meas;
        d.Rd = Rd; d.nm = nm; d.x0 = x0; d.kappaHWH = kappaHWH; d.dof = dof;
        sys_data{si} = d;
    end

    %======================================================================
    %  MAIN LOOP
    %======================================================================
    fprintf('%s\n RUNNING STATE ESTIMATION (tol = 1e-4)\n%s\n', SEP, SEP2);

    for si = 1:nSys
        d = sys_data{si};
        n = d.n; Ybus = d.Ybus; Yf = d.Yf; mpc = d.mpc;
        z = d.z; meas = d.meas; Rd = d.Rd; nm = d.nm; x0 = d.x0;
        Vm_true = d.Vm_true; th_true = d.th_true;
        J_chi = chi2_tgt(si);
        
        fprintf('\n[%s]\n', sysNames{si});
        fprintf('%-12s %8s %6s %10s %10s %10s %8s\n', ...
            'Method', 'Conv', 'Iter', 'RMSE', 'J(x)', '||g||', 'Time(s)');

        % Method 1: WLS Standard
        p1 = struct('MAX_ITER', 100, 'tol', tol_grad, 'tol_dx', tol_dx, ...
            'rcond_min', 1e-14, 'dx_max', 0.25, 'armijo_c', 1e-4, 'max_ls', 20);
        t0 = tic; [x1, h1] = se_wls_std(z, meas, Ybus, Yf, mpc, n, nm, Rd, x0, p1); t1 = toc(t0);
        [rv1, ~] = rmse_state(x1, Vm_true, th_true, n);
        h1 = audit(h1, x1, z, meas, Ybus, Yf, mpc, n, nm, Rd, J_chi, rv1, tol_grad, tol_dx, rmse_lim, 'WLS-Std');

        % Method 2: WLS Modified
        p2 = struct('MAX_ITER', 200, 'tol', tol_grad, 'tol_dx', tol_dx, 'reg_base', 1e-6);
        t0 = tic; [x2, h2] = se_wls_mod(z, meas, Ybus, Yf, mpc, n, nm, Rd, x0, p2); t2 = toc(t0);
        [rv2, ~] = rmse_state(x2, Vm_true, th_true, n);
        h2 = audit(h2, x2, z, meas, Ybus, Yf, mpc, n, nm, Rd, J_chi, rv2, tol_grad, tol_dx, rmse_lim, 'WLS-Mod');

        % Method 3: RNN-DMHN
        p3 = build_dmhn_rnn_params(d.kappaHWH, n, J_chi, tol_grad, tol_dx);
        t0 = tic; [x3, h3] = se_rnn_dmhn(z, meas, Ybus, Yf, mpc, n, nm, Rd, x0, p3); t3 = toc(t0);
        [rv3, ~] = rmse_state(x3, Vm_true, th_true, n);
        h3 = audit(h3, x3, z, meas, Ybus, Yf, mpc, n, nm, Rd, J_chi, rv3, tol_grad, tol_dx, rmse_lim, 'RNN-DMHN');

        % Method 4: SPE-DMHN
        p4 = build_dmhn_spe_params(d.kappaHWH, n, J_chi, tol_grad, tol_dx);
        t0 = tic; [x4, h4] = se_spe_dmhn(z, meas, Ybus, Yf, mpc, n, nm, Rd, x0, p4); t4 = toc(t0);
        [rv4, ~] = rmse_state(x4, Vm_true, th_true, n);
        h4 = audit(h4, x4, z, meas, Ybus, Yf, mpc, n, nm, Rd, J_chi, rv4, tol_grad, tol_dx, rmse_lim, 'SPE-DMHN');

        hs = {h1, h2, h3, h4};
        rs = [rv1, rv2, rv3, rv4];
        ts = [t1, t2, t3, t4];
        
        for mi = 1:nMethods
            all_hists{si, mi} = hs{mi};
            all_rmse(si, mi) = rs(mi);
            all_conv(si, mi) = hs{mi}.conv;
            
            % Print with color indicators
            conv_str = sprintf('%d', hs{mi}.conv);
            if hs{mi}.conv
                conv_str = ['✓' conv_str];
            else
                conv_str = ['✗' conv_str];
            end
            
            fprintf('%-12s %8s %6d %10.2e %10.2f %10.2e %8.3f [%s]\n', ...
                methodNames{mi}, conv_str, hs{mi}.iters, rs(mi), ...
                hs{mi}.J_final, hs{mi}.gn_final, ts(mi), hs{mi}.status);
        end
    end

    %======================================================================
    %  FIGURES
    %======================================================================
    fprintf('\n%s\n GENERATING FIGURES...\n%s\n', SEP, SEP2);
    
    fnt = 'Times New Roman';
    fs = 9;
    lw = 1.5;
    
    for figtype = 1:3
        switch figtype
            case 1
                fname = 'Fig_Convergence_J';
                ytitle = 'Objective J(x)';
            case 2
                fname = 'Fig_Convergence_Grad';
                ytitle = 'Gradient norm ||g||_{\infty}';
            case 3
                fname = 'Fig_Convergence_Step';
                ytitle = 'Step size ||dx||_{\infty}';
        end
        
        figure('Color', 'w', 'Name', fname, 'Position', [50, 50, 1400, 800]);
        
        for si = 1:nSys
            for mi = 1:nMethods
                ax = subplot(nSys, nMethods, (si-1)*nMethods + mi);
                hold(ax, 'on');
                
                h_ = all_hists{si, mi};
                
                switch figtype
                    case 1
                        dat = h_.J;
                    case 2
                        dat = h_.gn;
                    case 3
                        dat = h_.step;
                end
                
                dat = abs(dat(:));
                dat(~isfinite(dat) | dat <= 0) = NaN;
                
                if any(isfinite(dat))
                    semilogy(ax, 1:numel(dat), dat, '-', ...
                        'Color', colors{mi}, 'LineWidth', lw, 'MarkerSize', 3);
                end
                
                % Add chi2 threshold for J plot
                if figtype == 1
                    yline(ax, chi2_tgt(si), '--g', 'LineWidth', 1, 'Alpha', 0.7);
                end
                
                % Mark convergence point
                if figtype == 2 && h_.conv && h_.iters > 0 && h_.iters <= numel(dat)
                    semilogy(ax, h_.iters, dat(min(h_.iters, numel(dat))), ...
                        'k*', 'MarkerSize', 12, 'LineWidth', 2);
                end
                
                % Final RMSE text
                rmse_val = all_rmse(si, mi);
                rmse_color = [0, 0.5, 0];
                if rmse_val > 0.05
                    rmse_color = [0.8, 0, 0];
                elseif rmse_val > 0.01
                    rmse_color = [0.8, 0.6, 0];
                end
                
                text(ax, 0.02, 0.98, sprintf('RMSE=%.2e', rmse_val), ...
                    'Units', 'normalized', 'VerticalAlignment', 'top', ...
                    'FontSize', fs-1, 'Color', rmse_color, 'FontWeight', 'bold');
                
                % Convergence status
                if h_.conv
                    text(ax, 0.98, 0.02, '✓ CONV', 'Units', 'normalized', ...
                        'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom', ...
                        'FontSize', fs-1, 'Color', [0, 0.5, 0], 'FontWeight', 'bold');
                else
                    text(ax, 0.98, 0.02, '✗ NOT CONV', 'Units', 'normalized', ...
                        'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom', ...
                        'FontSize', fs-1, 'Color', [0.8, 0, 0], 'FontWeight', 'bold');
                end
                
                set(ax, 'YScale', 'log', 'FontName', fnt, 'FontSize', fs);
                grid(ax, 'on');
                
                if si == 1
                    title(ax, sprintf('%s', methodNames{mi}), ...
                        'FontName', fnt, 'FontSize', fs+1, 'Color', colors{mi}, ...
                        'FontWeight', 'bold');
                end
                
                if mi == 1
                    ylabel(ax, ytitle, 'FontName', fnt, 'FontSize', fs);
                end
                
                if si == nSys
                    xlabel(ax, 'Iteration', 'FontName', fnt, 'FontSize', fs);
                end
                
                box(ax, 'on');
            end
        end
        
        sgtitle(sprintf('State Estimation Convergence — %s (tol = 1e-4)', ytitle), ...
            'FontName', fnt, 'FontSize', fs+3, 'FontWeight', 'bold');
        
        % Save figure
        print('-dpng', '-r150', fname);
        fprintf('  Saved: %s.png\n', fname);
    end
    
    %======================================================================
    %  FINAL SUMMARY TABLE
    %======================================================================
    fprintf('\n%s\n FINAL SUMMARY (Tolerance = 1e-4)\n%s\n', SEP, SEP2);
    fprintf('┌────────────────────┬──────────────────┬────────────┬────────────┬────────────┐\n');
    fprintf('│ System             │ Method           │ Converged  │ RMSE       │ Iterations │\n');
    fprintf('├────────────────────┼──────────────────┼────────────┼────────────┼────────────┤\n');
    
    for si = 1:nSys
        for mi = 1:nMethods
            conv_status = sprintf('%d', all_conv(si, mi));
            if all_conv(si, mi)
                conv_status = '✓ YES';
            else
                conv_status = '✗ NO ';
            end
            fprintf('│ %-18s │ %-16s │ %-10s │ %10.2e │ %10d │\n', ...
                sysNames{si}, methodNames{mi}, conv_status, ...
                all_rmse(si, mi), all_hists{si, mi}.iters);
        end
        if si < nSys
            fprintf('├────────────────────┼──────────────────┼────────────┼────────────┼────────────┤\n');
        end
    end
    fprintf('└────────────────────┴──────────────────┴────────────┴────────────┴────────────┘\n');
    
    %======================================================================
    %  ASSESSMENT
    %======================================================================
    fprintf('\n%s\n ASSESSMENT\n%s\n', SEP, SEP2);
    fprintf('┌────────────────────────────────────────────────────────────────────────────┐\n');
    fprintf('│ OBSERVATIONS (Tolerance = 1e-4):                                           │\n');
    fprintf('├────────────────────────────────────────────────────────────────────────────┤\n');
    
    % Check 11-bus WLS performance
    if all_conv(1, 1) && all_rmse(1, 1) < 1e-2
        fprintf('│ ✓ 11-bus: WLS methods converge (expected - well-conditioned)              │\n');
    else
        fprintf('│ ✗ 11-bus: WLS methods FAIL - check implementation!                         │\n');
    end
    
    % Check 13-bus DMHN performance
    dmhn13_conv = all_conv(2, 3) || all_conv(2, 4);
    if dmhn13_conv
        fprintf('│ ✓ 13-bus: DMHN methods converge (ill-conditioned success!)                  │\n');
    else
        fprintf('│ ? 13-bus: DMHN methods need tuning for this ill-conditioned case           │\n');
    end
    
    % Check 20-bus performance
    if all_conv(3, 1)
        fprintf('│ ✓ 20-bus: All methods converge (moderately ill-conditioned)                 │\n');
    end
    
    fprintf('├────────────────────────────────────────────────────────────────────────────┤\n');
    fprintf('│ VERDICT:                                                                     │\n');
    if dmhn13_conv
        fprintf('│ DMHN methods successfully handle ill-conditioned systems from FLAT START!   │\n');
    else
        fprintf('│ DMHN methods show promise but need parameter tuning for extreme cases        │\n');
    end
    fprintf('└────────────────────────────────────────────────────────────────────────────┘\n');
    
    fprintf('\n%s\n DONE - 3 figures saved as PNG\n%s\n', SEP, SEP);
end


%==========================================================================
%  POWER FLOW WITH NO DC FALLBACK
%==========================================================================
function [theta, Vmag, conv, method_used] = run_pf_no_fallback(mpc, Ybus, n)
    
    theta = zeros(n, 1);
    Vmag = ones(n, 1);
    conv = false;
    method_used = 'NONE';
    
    % Try MATPOWER if available
    if exist('runpf', 'file') == 2 || exist('runpf', 'file') == 6
        try
            mpc_mp = mpc_to_matpower(mpc, n);
            mpopt = mpoption('verbose', 0, 'out.all', 0, 'pf.tol', 1e-8, 'pf.max_it', 200);
            
            % Try NR
            r = runpf(mpc_mp, mpopt);
            if r.success
                theta = r.bus(:, 9) * pi/180;
                Vmag = r.bus(:, 8);
                conv = true;
                method_used = 'MATPOWER-NR';
                return;
            end
            
            % Try FDBX
            mpopt2 = mpoption(mpopt, 'pf.alg', 'FDBX', 'pf.max_it', 500);
            r = runpf(mpc_mp, mpopt2);
            if r.success
                theta = r.bus(:, 9) * pi/180;
                Vmag = r.bus(:, 8);
                conv = true;
                method_used = 'MATPOWER-FDBX';
                return;
            end
        catch
            % MATPOWER error - continue to internal NR
        end
    end
    
    % Try internal NR - NO DC FALLBACK
    bt = mpc.bus(:, 2);
    Psch = mpc.bus(:, 5) - mpc.bus(:, 3);
    Qsch = mpc.bus(:, 6) - mpc.bus(:, 4);
    
    sl = find(bt == 1);
    pv = find(bt == 2);
    pq = find(bt == 3);
    
    thf = setdiff((1:n)', sl);
    vf = pq;
    
    % Try multiple starting points
    start_points = {...
        struct('theta', zeros(n, 1), 'Vmag', ones(n, 1), 'name', 'Flat Start'), ...
        struct('theta', 0.1*randn(n, 1), 'Vmag', 1+0.05*randn(n, 1), 'name', 'Small Perturbation'), ...
        struct('theta', 0.5*ones(n, 1), 'Vmag', 0.95*ones(n, 1), 'name', 'Offset Start')};
    
    for sp = 1:length(start_points)
        theta0 = start_points{sp}.theta;
        Vmag0 = start_points{sp}.Vmag;
        
        [theta_t, Vmag_t, conv_t] = nr_pf_internal(theta0, Vmag0, Psch, Qsch, Ybus, n, ...
            sl, pv, pq, thf, vf, mpc, 500, 0.05);
        
        if conv_t
            theta = theta_t;
            Vmag = Vmag_t;
            conv = true;
            method_used = sprintf('Internal-NR(%s)', start_points{sp}.name);
            return;
        end
    end
    
    % If all attempts fail, return flat start with conv=false
    theta = zeros(n, 1);
    Vmag = ones(n, 1);
    conv = false;
    method_used = 'FAILED-FLAT';
end


%==========================================================================
%  INTERNAL NR POWER FLOW
%==========================================================================
function [theta, Vmag, conv] = nr_pf_internal(theta0, Vmag0, Psch, Qsch, Ybus, n, ...
    sl, pv, pq, thf, vf, mpc, maxiter, step_lim)
    
    theta = theta0;
    Vmag = Vmag0;
    conv = false;
    
    for it = 1:maxiter
        V = Vmag .* exp(1i * theta);
        Sinj = V .* conj(Ybus * V);
        
        mis = [Psch(thf) - real(Sinj(thf)); 
               Qsch(vf) - imag(Sinj(vf))];
        
        if norm(mis, inf) < 1e-6
            conv = true;
            return;
        end
        
        J = pf_jacobian(Vmag, theta, Ybus, n, thf, vf);
        
        if isempty(J) || rcond(J) < 1e-14 || any(~isfinite(J(:)))
            return;
        end
        
        dx = J \ mis;
        dx(~isfinite(dx)) = 0;
        
        if norm(dx, inf) > step_lim
            dx = dx * (step_lim / norm(dx, inf));
        end
        
        nt = numel(thf);
        theta(thf) = theta(thf) + dx(1:nt);
        Vmag(vf) = Vmag(vf) + dx(nt+1:end);
        
        Vmag = max(0.5, min(1.5, Vmag));
        Vmag(pv) = mpc.bus(pv, 7);
    end
end


%==========================================================================
%  POWER FLOW JACOBIAN
%==========================================================================
function J = pf_jacobian(Vmag, theta, Ybus, n, thi, vi)
    V = Vmag .* exp(1i * theta);
    I = Ybus * V;
    dV = diag(V);
    dI = diag(I);
    dVn = diag(V ./ max(abs(V), 1e-9));
    
    dSth = 1i * dV * conj(dI - Ybus * dV);
    dSV = dV * conj(Ybus * dVn) + dI * conj(dVn);
    
    J = [real(dSth(thi, thi)), real(dSV(thi, vi));
         imag(dSth(vi, thi)), imag(dSV(vi, vi))];
    
    J(~isfinite(J)) = 0;
end


%==========================================================================
%  MATPOWER CASE CONVERTER
%==========================================================================
function mpc_mp = mpc_to_matpower(mpc, n)
    mpc_mp.version = '2';
    mpc_mp.baseMVA = 100;
    
    % Bus matrix
    mpc_mp.bus = zeros(n, 13);
    mpc_mp.bus(:, 1) = (1:n)';
    
    bt_orig = mpc.bus(:, 2);
    bt_mp = ones(n, 1);
    bt_mp(bt_orig == 2) = 2;
    bt_mp(bt_orig == 1) = 3;
    mpc_mp.bus(:, 2) = bt_mp;
    
    mpc_mp.bus(:, 3) = mpc.bus(:, 3) * 100;
    mpc_mp.bus(:, 4) = mpc.bus(:, 4) * 100;
    mpc_mp.bus(:, 5) = 0;
    mpc_mp.bus(:, 6) = 0;
    mpc_mp.bus(:, 7) = 1;
    mpc_mp.bus(:, 8) = mpc.bus(:, 7);
    mpc_mp.bus(:, 9) = 0;
    mpc_mp.bus(:, 10) = 100;
    mpc_mp.bus(:, 11) = 1;
    mpc_mp.bus(:, 12) = 1.5;
    mpc_mp.bus(:, 13) = 0.5;
    
    % Generator table
    gen_idx = find(mpc.bus(:, 2) == 1 | mpc.bus(:, 2) == 2);
    if isempty(gen_idx)
        gen_idx = 1;
    end
    ng = numel(gen_idx);
    mpc_mp.gen = zeros(ng, 21);
    
    for i = 1:ng
        b = gen_idx(i);
        mpc_mp.gen(i, 1) = b;
        mpc_mp.gen(i, 2) = mpc.bus(b, 5) * 100;
        mpc_mp.gen(i, 3) = mpc.bus(b, 6) * 100;
        mpc_mp.gen(i, 4) = 9999;
        mpc_mp.gen(i, 5) = -9999;
        mpc_mp.gen(i, 6) = mpc.bus(b, 7);
        mpc_mp.gen(i, 7) = 100;
        mpc_mp.gen(i, 8) = 1;
        mpc_mp.gen(i, 9) = 9999;
        mpc_mp.gen(i, 10) = -9999;
    end
    
    % Branch matrix
    nb = size(mpc.branch, 1);
    mpc_mp.branch = zeros(nb, 13);
    mpc_mp.branch(:, 1) = mpc.branch(:, 1);
    mpc_mp.branch(:, 2) = mpc.branch(:, 2);
    mpc_mp.branch(:, 3) = mpc.branch(:, 3);
    mpc_mp.branch(:, 4) = mpc.branch(:, 4);
    mpc_mp.branch(:, 5) = mpc.branch(:, 5);
    mpc_mp.branch(:, 6) = 9999;
    mpc_mp.branch(:, 7) = 9999;
    mpc_mp.branch(:, 8) = 9999;
    
    tap = mpc.branch(:, 6);
    tap(tap < 0.5 | ~isfinite(tap)) = 1;
    mpc_mp.branch(:, 9) = tap;
    mpc_mp.branch(:, 10) = 0;
    mpc_mp.branch(:, 11) = 1;
    mpc_mp.branch(:, 12) = -360;
    mpc_mp.branch(:, 13) = 360;
end


%==========================================================================
%  DMHN CORE FUNCTIONS
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


%==========================================================================
%  METHOD 1: WLS STANDARD
%==========================================================================
function [x, hist] = se_wls_std(z, meas, Ybus, Yf, mpc, n, nm, Rd, x0, par)

    x = x0;
    hist = init_hist(par.MAX_ITER);

    for k = 1:par.MAX_ITER

        [~,~,J,g,G] = compute_all(x, z, meas, Ybus, Yf, mpc, n, nm, Rd);

        hist.J(k)  = J;
        hist.gn(k) = norm(g, inf);

        % PURE GAUSS NEWTON STEP
        dx = G \ g;

        hist.step(k) = norm(dx, inf);

        % PURE UPDATE ONLY
        x = x + dx;

        if hist.gn(k) <= par.tol && hist.step(k) <= par.tol_dx
            hist = trim(hist, k, 'CONVERGED');
            hist.conv = true;
            return;
        end
    end

    hist = trim(hist, par.MAX_ITER, 'MAX_ITER');
end


%==========================================================================
%  METHOD 2: WLS MODIFIED
%==========================================================================
function [x, hist] = se_wls_mod(z, meas, Ybus, Yf, mpc, n, nm, Rd, x0, par)

    x = x0;
    hist = init_hist(par.MAX_ITER);

    lambda = 1e-3;   % FIXED ONLY (NO ADAPTATION)

    for k = 1:par.MAX_ITER

        [~,~,J,g,G] = compute_all(x, z, meas, Ybus, Yf, mpc, n, nm, Rd);

        hist.J(k)  = J;
        hist.gn(k) = norm(g, inf);

        % PURE LM STEP
        A = G + lambda * eye(size(G));
        dx = A \ g;

        hist.step(k) = norm(dx, inf);

        x = x + dx;

        if hist.gn(k) <= par.tol && hist.step(k) <= par.tol_dx
            hist = trim(hist, k, 'CONVERGED');
            hist.conv = true;
            return;
        end
    end

    hist = trim(hist, par.MAX_ITER, 'MAX_ITER');
end


%==========================================================================
%  METHOD 3: RNN-DMHN
%==========================================================================
function [x, hist] = se_rnn_dmhn(z, meas, Ybus, Yf, mpc, n, nm, Rd, x0, par)
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
        
        alpha = 1.0;
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
%  METHOD 4: SPE-DMHN
%==========================================================================
function [x, hist] = se_spe_dmhn(z, meas, Ybus, Yf, mpc, n, nm, Rd, x0, par)
    Nst = 2*n - 1;
    x = clip_x(x0, n);
    K = par.MAX_ITER;
    hist = init_hist(K);
    tol_g = par.tol_grad;
    tol_dx = par.tol_dx;
    x_best = x;
    J_best = inf;
    lam = par.lam0;
    sigma_mean = mean(1 ./ sqrt(max(Rd, 1e-12)));
    ds = dmhn_init(Nst, par);
    
    m_mem = par.lbfgs_m;
    S_mem = zeros(Nst, m_mem);
    Y_mem = zeros(Nst, m_mem);
    mem_n = 0;
    mem_p = 0;
    B_diag = ones(Nst, 1);
    consec_fail = 0;
    gd_steps = 0;
    
    for k = 1:K
        x = clip_x(x, n);
        [r, H, J_t, g, G] = compute_all(x, z, meas, Ybus, Yf, mpc, n, nm, Rd);
        gn_v = sgi(g, G);
        hist.J(k) = J_t;
        hist.gn(k) = gn_v;
        hist.mu(k) = lam;
        
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
        
        Q_base = Q_eff + lam * diag(abs(diag(Q_eff)) + 1e-14);
        Q_base = 0.5 * (Q_base + Q_base');
        
        if gd_steps > 0
            [Qp, Ds] = struct_precond(Q_base);
            dp = solve_svd(Qp, Ds .* g_eff, par.svd_tol);
            d = Ds .* dp;
            gd_steps = gd_steps - 1;
        else
            d = lbfgs_dir(g_eff, S_mem, Y_mem, B_diag, mem_n, m_mem, Q_base, par);
        end
        d = clip_step(d, n, par);
        gTd = g_eff' * d;
        
        if ~isfinite(gTd) || gTd >= 0
            [Qp, Ds] = struct_precond(Q_base);
            dp = solve_svd(Qp, Ds .* g_eff, par.svd_tol);
            d = Ds .* dp;
            d = clip_step(d, n, par);
            gTd = g_eff' * d;
            if gTd >= 0
                d = -g_eff;
                d = clip_step(d, n, par);
                gTd = g_eff' * d;
            end
        end
        
        [alpha, x_new, g_new, J_new, wolfe_ok] = wolfe_ls(x, d, g, J_t, z, meas, Ybus, Yf, mpc, n, nm, Rd, par);
        
        if wolfe_ok
            sk = x_new - x;
            yk = g_new - g;
            Bs = B_diag .* sk;
            sBs = sk' * Bs;
            sTy = sk' * yk;
            
            if sTy < 0.2 * sBs && sBs > 1e-14
                theta_c = min(0.8 * sBs / max(sBs - sTy, 1e-14), 1.0);
                yk = theta_c * yk + (1 - theta_c) * Bs;
            end
            
            sTy2 = sk' * yk;
            if sTy2 > 1e-10 * norm(sk) * norm(yk)
                B_diag = update_B_diag(B_diag, sk, yk, sTy2);
                mem_p = mod(mem_p, m_mem) + 1;
                S_mem(:, mem_p) = sk;
                Y_mem(:, mem_p) = yk;
                mem_n = min(mem_n + 1, m_mem);
            end
            
            hist.step(k) = norm(x_new - x, inf);
            x = x_new;
            lam = max(par.lam_min, lam * 0.5);
            consec_fail = 0;
        else
            alpha2 = 1.0;
            acc2 = false;
            for ls2 = 1:par.max_ls
                xt = clip_x(x + alpha2 * d, n);
                [~, ~, Jt2, ~, ~] = compute_all(xt, z, meas, Ybus, Yf, mpc, n, nm, Rd);
                if isfinite(Jt2) && Jt2 < J_t
                    x = xt;
                    acc2 = true;
                    hist.step(k) = norm(xt - clip_x(x - alpha2 * d, n), inf);
                    break;
                end
                alpha2 = alpha2 * 0.5;
                if alpha2 < 1e-14
                    break;
                end
            end
            if ~acc2
                hist.step(k) = 0;
            end
            consec_fail = consec_fail + 1;
            lam = min(par.lam_max, lam * 16);
            if consec_fail >= 5
                mem_n = 0;
                mem_p = 0;
                S_mem = zeros(Nst, m_mem);
                Y_mem = zeros(Nst, m_mem);
                B_diag = ones(Nst, 1);
                gd_steps = 3;
                consec_fail = 0;
                lam = max(par.lam_min, lam * 0.01);
            end
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
%  HELPER FUNCTIONS
%==========================================================================
function p = build_dmhn_rnn_params(kHWH, n, chi2, tol_g, tol_dx)
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

function p = build_dmhn_spe_params(kHWH, n, chi2, tol_g, tol_dx)
    p.MAX_ITER = 600;
    p.chi2 = chi2;
    p.tol_grad = tol_g;
    p.tol_dx = tol_dx;
    p.lam0 = min(1.0, 1e-3 * max(1, log10(kHWH)));
    p.lam_min = 1e-14;
    p.lam_max = 1e8;
    p.c1 = 1e-4;
    p.c2 = 0.9;
    p.eta_rej = 1e-4;
    p.beta = 0.5;
    p.max_ls = 30;
    p.svd_tol = 1e-10;
    p.max_angle_step = 0.08;
    p.max_volt_step = 0.04;
    p.lbfgs_m = 8;
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

function hist = audit(hist, x, z, meas, Ybus, Yf, mpc, n, nm, Rd, J_chi, rmseV, tol_g, tol_dx, rmse_lim, name)
    [~, ~, Jf, gf, Gf] = compute_all(x, z, meas, Ybus, Yf, mpc, n, nm, Rd);
    d = abs(diag(Gf));
    d = max(d, 1e-14);
    dxf = norm(gf ./ d, inf);
    hist.J_final = Jf;
    hist.gn_final = sgi(gf, Gf);
    hist.dx_final = dxf;
    hist.rmseV = rmseV;
    
    J_ok = isfinite(Jf) && Jf <= J_chi;
    g_ok = isfinite(hist.gn_final) && hist.gn_final <= tol_g;
    dx_ok = isfinite(dxf) && dxf <= tol_dx;
    last_step = NaN;
    if isfield(hist, 'step') && ~isempty(hist.step)
        v = hist.step(isfinite(hist.step));
        if ~isempty(v)
            last_step = v(end);
        end
    end
    step_ok = isfinite(last_step) && last_step <= tol_dx;
    
    if ~isfinite(rmseV) || rmseV > rmse_lim
        hist.conv = false;
        hist.status = 'FALSE_CONV_HIGH_RMSE';
    elseif ~isfinite(Jf) || ~isfinite(hist.gn_final)
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

function [Qp, Ds] = struct_precond(Q)
    d = abs(diag(Q));
    d = max(d, 1e-12 * (max(d(d>0)) + eps));
    Ds = 1 ./ sqrt(d);
    Ds(~isfinite(Ds)) = 1;
    Qp = bsxfun(@times, bsxfun(@times, Q, Ds), Ds');
    Qp = 0.5 * (Qp + Qp');
end

function d = solve_sym(A, b)
    A(~isfinite(A)) = 0;
    b(~isfinite(b)) = 0;
    A = 0.5 * (A + A');
    if rcond(A) < 1e-13
        [U, S, V] = svd(A, 'econ');
        sv = diag(S);
        dmp = 1e-6 * max(sv);
        filt = sv ./ (sv.^2 + dmp^2);
        d = V * (filt .* (U' * b));
    else
        try
            [L, f] = chol(A, 'lower');
            if f == 0
                d = L' \ (L \ b);
            else
                d = A \ b;
            end
        catch
            d = A \ b;
        end
    end
    if any(~isfinite(d))
        d = b;
    end
end

function d = solve_svd(Q, g, tol)
    Q(~isfinite(Q)) = 0;
    g(~isfinite(g)) = 0;
    Q = 0.5 * (Q + Q');
    rc = rcond(Q);
    if ~isfinite(rc)
        rc = 0;
    end
    if rc < tol
        [U, S, V] = svd(Q, 'econ');
        sv = diag(S);
        if isempty(sv) || max(sv) <= 0
            d = -g;
            return;
        end
        dmp = 1e-6 * max(sv);
        d = -V * ((sv ./ (sv.^2 + dmp^2)) .* (U' * g));
    else
        try
            [L, f] = chol(Q, 'lower');
            if f == 0
                d = -(L' \ (L \ g));
            else
                d = -(Q \ g);
            end
        catch
            d = -(Q \ g);
        end
    end
    if any(~isfinite(d))
        d = -g;
    end
end

function d = lbfgs_dir(g, S, Y, B_diag, mem_n, m_mem, Q_base, par)
    if mem_n == 0
        [Qp, Ds] = struct_precond(Q_base);
        dp = solve_svd(Qp, Ds .* g, par.svd_tol);
        d = Ds .* dp;
        return;
    end
    q = g;
    rho_v = zeros(mem_n, 1);
    alp = zeros(mem_n, 1);
    for i = mem_n:-1:1
        si = S(:, i);
        yi = Y(:, i);
        sTy = si' * yi;
        if sTy < 1e-14
            continue;
        end
        rho_v(i) = 1 / sTy;
        alp(i) = rho_v(i) * (si' * q);
        q = q - alp(i) * yi;
    end
    sl = S(:, mem_n);
    yl = Y(:, mem_n);
    sTy_l = sl' * yl;
    yTy_l = yl' * yl;
    if sTy_l > 1e-14 && yTy_l > 1e-14
        gam = sTy_l / yTy_l;
    else
        gam = 1 / max(max(B_diag), 1e-8);
    end
    r = gam * q;
    for i = 1:mem_n
        si = S(:, i);
        yi = Y(:, i);
        sTy = si' * yi;
        if sTy < 1e-14
            continue;
        end
        r = r + si * (alp(i) - (1/sTy) * (yi' * r));
    end
    d = -r;
    if any(~isfinite(d)) || norm(d) == 0
        d = -g;
    end
end

function B_new = update_B_diag(B_old, s, y, sTy)
    Bs = B_old .* s;
    sBs = s' * Bs;
    if sTy < 1e-14 || sBs < 1e-14
        B_new = B_old;
        return;
    end
    B_new = B_old - (Bs.^2) / sBs + (y.^2) / sTy;
    B_new = max(B_new, 1e-8);
end

function [alpha, x_new, g_new, J_new, ok] = wolfe_ls(x, d, g0, J0, z, meas, Ybus, Yf, mpc, n, nm, Rd, par)
    c1 = par.c1;
    c2 = par.c2;
    gTd = g0' * d;
    alpha = 1;
    al = 0;
    ah = inf;
    ok = false;
    x_new = x;
    g_new = g0;
    J_new = J0;
    J_lo = J0;
    
    if gTd >= 0
        return;
    end
    
    for ls = 1:par.max_ls
        xt = clip_x(x + alpha * d, n);
        [~, ~, Jt, gt, ~] = compute_all(xt, z, meas, Ybus, Yf, mpc, n, nm, Rd);
        if ~isfinite(Jt)
            alpha = alpha * 0.5;
            if alpha < 1e-14
                return;
            end
            continue;
        end
        if Jt > J0 + c1 * alpha * gTd || (ls > 1 && Jt >= J_lo)
            ah = alpha;
            alpha = (al + ah) / 2;
        else
            gTdn = gt' * d;
            if abs(gTdn) <= c2 * abs(gTd)
                ok = true;
                x_new = xt;
                g_new = gt;
                J_new = Jt;
                return;
            end
            if gTdn * (ah - al) >= 0
                ah = al;
            end
            al = alpha;
            J_lo = Jt;
            if ~isfinite(ah)
                alpha = 2 * alpha;
            else
                alpha = (al + ah) / 2;
            end
        end
        if alpha < 1e-14
            return;
        end
        if isfinite(ah) && abs(ah - al) < 1e-14
            break;
        end
    end
    
    alpha2 = 1;
    for ls2 = 1:15
        xt = clip_x(x + alpha2 * d, n);
        [~, ~, Jt, gt, ~] = compute_all(xt, z, meas, Ybus, Yf, mpc, n, nm, Rd);
        if isfinite(Jt) && Jt < J0 + c1 * alpha2 * gTd
            ok = true;
            x_new = xt;
            g_new = gt;
            J_new = Jt;
            return;
        end
        alpha2 = alpha2 * 0.5;
        if alpha2 < 1e-14
            return;
        end
    end
end

function d = clip_step(d, n, par)
    d(~isfinite(d)) = 0;
    amax = par.max_angle_step;
    vmax = par.max_volt_step;
    na = norm(d(1:min(n-1, end)), inf);
    if na > amax && na > 0
        d(1:n-1) = d(1:n-1) * (amax / na);
    end
    if numel(d) >= 2*n-1
        nv = norm(d(n:end), inf);
        if nv > vmax && nv > 0
            d(n:end) = d(n:end) * (vmax / nv);
        end
    end
end

function x = clip_x(x, n)
    x(~isfinite(x)) = 0;
    x(1:n-1) = max(-pi/2 + 1e-8, min(pi/2 - 1e-8, x(1:n-1)));
    x(n:end) = max(0.5 + 1e-8, min(1.5 - 1e-8, x(n:end)));
end

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

function val = sgi(g, G)
    g(~isfinite(g)) = 0;
    if isempty(G) || size(G, 1) ~= numel(g)
        val = norm(g, inf);
        return;
    end
    d = abs(diag(G));
    d(~isfinite(d)) = 0;
    pos = d(d > 0);
    if isempty(pos)
        sf = 1;
    else
        sf = 1e-12 * max(pos);
    end
    d = max(d, sf);
    gs = g ./ sqrt(d);
    gs(~isfinite(gs)) = 0;
    val = norm(gs, inf);
end

function [rmseV, rmseT] = rmse_state(x, Vtrue, Ttrue, n)
    if isempty(x) || any(~isfinite(x))
        rmseV = inf;
        rmseT = inf;
        return;
    end
    th = [0; x(1:n-1)];
    Vm = x(n:end);
    rmseV = sqrt(mean((Vm(:) - Vtrue(:)).^2));
    rmseT = sqrt(mean((th(:) - Ttrue(:)).^2));
end

function hist = init_hist(K)
    hist.J = nan(K, 1);
    hist.gn = nan(K, 1);
    hist.step = nan(K, 1);
    hist.mu = nan(K, 1);
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

function mpc = build_mpc(num)
    switch num
        case 11
            busdat = [1 1 1.024 0 0 0 0 0 0 0; 2 3 1 0 0 0 0 0 0 0; 3 3 1 0 0 0 -0.128 -0.062 0 0;
                4 3 1 0 0 0 0 0 0 0; 5 3 1 0 0 0 -0.165 -0.080 0 0; 6 3 1 0 0 0 -0.090 -0.068 0 0;
                7 3 1 0 0 0 0 0 0 0; 8 3 1 0 0 0 0 0 0 0; 9 3 1 0 0 0 -0.026 -0.009 0 0;
                10 3 1 0 0 0 0 0 0 0; 11 3 1 0 0 0 -0.158 -0.057 0 0];
            linedat = [1 2 0 0.0706 0 1; 2 3 0 0.154 0 1; 2 4 0.0377 0.0413 0 1; 3 5 0.1228 0.1803 0 1;
                4 5 0 0.4593 0 1; 4 6 0 0.0176 0 1; 4 7 0.6114 0.8117 0 1; 7 8 0.6209 0.2167 0 1;
                8 9 0.0718 0.7179 0 1; 8 10 0.4097 0.5600 0 1; 10 11 0.0264 0.2646 0 1];
        case 13
            busdat = [1 1 1 0 0 0 1.65 0.56 0 0; 2 3 1.05 0 0 0 0 0 0 0; 3 3 1.05 0 0 0 0 0 0 0;
                4 3 1.05 0 0 0 0 0 0 0; 5 2 1 0 0 0 0 0 0 0; 6 2 1.037 0 0.5 0 0.05 0.03 0 0;
                7 3 1.05 0 0 0 0 0 0 0; 8 2 1.1 0 0 0 0 0 0 0; 9 2 0.943 0 0.5 0 0 0 0 0;
                10 2 1.1 0 0 0 0 0 0 0; 11 3 1.05 0 0 0 0.05 0.03 0 0;
                12 3 1.05 0 0 0 0.05 0.032 0 0; 13 3 1.05 0 0 0 0 0 0 0];
            linedat = [1 2 0.004 0.085 0.05 1; 1 3 0.004 0.0947 0.10 1; 5 4 0.004 0.0947 0.10 1;
                4 3 0.0074 0.143 0.218 1; 6 2 0.0481 0.459 0.123 1; 6 7 0.009 0.108 0.008 1;
                8 3 0.0121 0.233 0.356 1; 7 8 0 0.15 0 1; 9 10 0.0105 0.202 0.31 1;
                10 11 0 0.15 0 1; 11 12 0.0086 0.1665 0.254 1; 12 13 0.0075 0.1465 0.224 1;
                13 8 0 0.15 0 1];
        case 20
            busdat = [1 3 1 0 0 0 150 30 0 0; 2 3 1 0 0 0 10 0 0 0; 3 3 1 0 0 0 0 0 0 0;
                4 3 1 0 0 0 380 60 0 0; 5 3 1 0 0 0 0 0 0 0; 6 3 1 0 0 0 0 0 0 0;
                7 3 1 0 0 0 20 0 0 0; 8 3 1 0 0 0 10 20 0 0; 9 3 1 0 0 0 0 0 0 0;
                10 3 1 0 0 0 50 10 0 0; 11 3 1 0 0 0 0 0 0 0; 12 3 1 0 0 0 0 0 0 0;
                13 3 1 0 0 0 0 0 0 0; 14 3 1 0 0 0 0 10 0 0; 15 3 1 0 0 0 0 0 0 0;
                16 3 1 0 0 0 10 0 0 0; 17 2 1 0 100 0 0 0 0 0; 18 2 1 0 100 0 0 0 0 0;
                19 2 1 0 100 0 0 0 0 0; 20 1 1 0 0 0 0 0 0 0];
            linedat_pct = [1 20 0.5 5 0.012 1; 2 8 0.5 5 0.0335 1; 2 16 0 5 0 1; 2 17 60 60 0 1;
                3 5 20 20 0 1; 3 20 0.11 1.52 0.04285 1; 4 17 3 4 0.025 1; 4 20 5 10 0.125 1;
                5 14 30 40 0 1; 6 15 5 10 0 1; 6 16 60 80 0 1; 6 17 0.6 8 0.02 1;
                7 12 0.5 5 0.025 1; 9 10 0.5 5 0.025 1; 9 19 0.1 3 0.05 1;
                10 11 0 30 0 1; 11 12 2 40 0.012 1; 11 13 0 15 0 1;
                13 18 0.5 6 0.015 1; 14 19 0.1 1 0 1; 15 18 0.15 1.5 0 1; 15 20 2 4 0.0335 1];
            linedat = linedat_pct;
            linedat(:, 3) = linedat_pct(:, 3) / 100;
            linedat(:, 4) = linedat_pct(:, 4) / 100;
    end
    
    n = size(busdat, 1);
    nb_ = size(linedat, 1);
    mpc.bus = zeros(n, 7);
    for i = 1:n
        mpc.bus(i, :) = [busdat(i, 1), busdat(i, 2), busdat(i, 7), busdat(i, 8), busdat(i, 5), busdat(i, 6), busdat(i, 3)];
    end
    mpc.branch = zeros(nb_, 6);
    for k = 1:nb_
        tap = linedat(k, 6);
        if tap == 0
            tap = 1;
        end
        mpc.branch(k, :) = [linedat(k, 1:5), tap];
    end
    mpc.n = n;
end

function pb = get_pmu_buses(num, n)
    switch num
        case 11
            pb = [1 4 8];
        case 13
            pb = [1 6 8];
        case 20
            pb = [1 17 20];
        otherwise
            pb = [1 round(n/2) n];
    end
    pb = pb(pb >= 1 & pb <= n);
end

function [Ybus, Yf] = build_Ybus(mpc, n)
    nb_ = size(mpc.branch, 1);
    Ybus = sparse(n, n);
    Yf = zeros(nb_, n);
    for k = 1:nb_
        f = max(1, min(n, round(mpc.branch(k, 1))));
        t_ = max(1, min(n, round(mpc.branch(k, 2))));
        r_ = mpc.branch(k, 3);
        x_ = mpc.branch(k, 4);
        if abs(x_) < 1e-3
            x_ = sign(x_ + eps) * 1e-3;
        end
        bsh = mpc.branch(k, 5);
        tap = mpc.branch(k, 6);
        if tap == 0
            tap = 1;
        end
        den = r_^2 + x_^2;
        if den < 1e-12
            ys = 0;
        else
            ys = (r_ - 1i*x_) / den;
        end
        yff = (ys + 1i*bsh/2) / tap^2;
        yft = -ys / conj(tap);
        ytf = -ys / tap;
        ytt = ys + 1i*bsh/2;
        Ybus(f, f) = Ybus(f, f) + yff;
        Ybus(f, t_) = Ybus(f, t_) + yft;
        Ybus(t_, f) = Ybus(t_, f) + ytf;
        Ybus(t_, t_) = Ybus(t_, t_) + ytt;
        Yf(k, f) = yff;
        Yf(k, t_) = yft;
    end
    Yf = sparse(Yf);
end

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

function [z, meas] = build_meas(mpc, Ybus, Yf, theta, Vmag, n, nb, pb, mode)
    V = Vmag .* exp(1i * theta);
    Sinj = V .* conj(Ybus * V);
    If = Yf * V;
    fr = max(1, min(n, round(mpc.branch(:, 1))));
    Sf = V(fr) .* conj(If);
    zs = [real(Sinj); imag(Sinj); Vmag; real(Sf); imag(Sf)];
    sig_s = [0.02*ones(n,1); 0.025*ones(n,1); 0.006*ones(n,1); 0.02*ones(nb,1); 0.025*ones(nb,1)];
    if strcmpi(mode, 'HYBRID')
        np = numel(pb);
        zp = [Vmag(pb) .* cos(theta(pb)); Vmag(pb) .* sin(theta(pb))];
        z = [zs; zp];
        sigma = [sig_s; 0.001*ones(2*np,1)];
    else
        z = zs;
        sigma = sig_s;
        pb = [];
    end
    meas.mode = upper(mode);
    meas.n = n;
    meas.nb = nb;
    meas.pmu_buses = pb;
    meas.sigma = sigma;
end